"""Local job queue: poll a text file of YAML config paths and run them
in parallel `turbigen` subprocesses. Exposed as `turbigen queue`."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

QUEUE_FILE_DEFAULT = "~/.turbigen/queue.txt"
PID_FILE_DEFAULT = "~/.turbigen/queue.pid"
POLL_INTERVAL_S = 5
VALID_SUFFIXES = {".yaml", ".yml"}

UNIT_TEMPLATE = """\
[Unit]
Description=turbigen local job queue
After=network.target

[Service]
Type=simple
WorkingDirectory={workdir}
ExecStart={uv} run turbigen queue --queue-file {queue_file}
Restart=on-failure
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=600

[Install]
WantedBy=default.target
"""


def _read_lines_locked(queue_file):
    """Read queue lines under an exclusive flock. Returns (fd, lines).
    Caller must close the fd to release the lock."""
    fd = os.open(queue_file, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    with os.fdopen(os.dup(fd), "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return fd, [ln for ln in lines if ln]


def _write_lines_atomic(queue_file, lines):
    """Atomically replace queue_file with the given lines."""
    tmp = queue_file.with_suffix(queue_file.suffix + ".tmp")
    with tmp.open("w") as f:
        for ln in lines:
            f.write(ln + "\n")
    os.replace(tmp, queue_file)


def _release(fd):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _dedupe(lines):
    seen = set()
    out = []
    for ln in lines:
        if ln in seen:
            continue
        seen.add(ln)
        out.append(ln)
    return out


def claim_next(queue_file):
    """Atomically pop the first line of the queue file (deduping as a
    side effect). Returns the popped line, or None if empty."""
    fd, lines = _read_lines_locked(queue_file)
    try:
        lines = _dedupe(lines)
        if not lines:
            _write_lines_atomic(queue_file, [])
            return None
        picked, *remaining = lines
        _write_lines_atomic(queue_file, remaining)
        return picked
    finally:
        _release(fd)


def validate(raw):
    """Return a Path if `raw` is an absolute path to an existing
    .yaml/.yml file. Returns None otherwise."""
    p = Path(raw)
    if not p.is_absolute():
        return None
    if not p.is_file():
        return None
    if p.suffix.lower() not in VALID_SUFFIXES:
        return None
    return p


def spawn(yaml_path):
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    binary = os.environ.get("TURBIGEN_BIN", "turbigen")
    return subprocess.Popen(
        [binary, "--no-job", str(yaml_path)],
        cwd=str(yaml_path.parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def serve(queue_file, max_workers, pid_file=None):
    queue_file = Path(queue_file)
    pid_file = Path(pid_file) if pid_file else Path(PID_FILE_DEFAULT).expanduser()
    running = {}
    state = {"stopping": False}

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{os.getpid()}\n")

    def _stop(signum, frame):
        if not state["stopping"]:
            print(f"signal {signum}: draining", flush=True)
        state["stopping"] = True
        for proc in list(running):
            proc.send_signal(signal.SIGTERM)

    def _hup(signum, frame):
        print("HUP cancel-all", flush=True)
        try:
            fd, _ = _read_lines_locked(queue_file)
            try:
                _write_lines_atomic(queue_file, [])
            finally:
                _release(fd)
        except OSError as e:
            print(f"HUP failed to truncate queue: {e}", flush=True)
        for proc in list(running):
            proc.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGHUP, _hup)

    print(f"Monitoring {queue_file} with {max_workers} workers.", flush=True)

    try:
        while True:
            for proc in list(running):
                rc = proc.poll()
                if rc is None:
                    continue
                path = running.pop(proc)
                tag = "DONE" if rc == 0 else "FAIL"
                print(f"{tag} {path}", flush=True)

            if state["stopping"]:
                if not running:
                    return
                time.sleep(0.2)
                continue

            while len(running) < max_workers:
                try:
                    line = claim_next(queue_file)
                except OSError as e:
                    print(f"queue read failed: {e}", flush=True)
                    break
                if line is None:
                    break
                resolved = validate(line)
                if resolved is None:
                    print(f"SKIP {line} (invalid)", flush=True)
                    continue
                proc = spawn(resolved)
                running[proc] = resolved
                print(f"START {resolved}", flush=True)

            time.sleep(POLL_INTERVAL_S)
    finally:
        try:
            pid_file.unlink()
        except FileNotFoundError:
            pass


def signal_daemon(sig=signal.SIGHUP, pid_file=None):
    """Send `sig` to the queue daemon if its pid file exists and the
    process is alive. Returns True if signalled, False otherwise."""
    pid_file = Path(pid_file) if pid_file else Path(PID_FILE_DEFAULT).expanduser()
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        try:
            pid_file.unlink()
        except FileNotFoundError:
            pass
        return False
    except PermissionError:
        return False


def _make_queue_parser():
    p = argparse.ArgumentParser(
        prog="turbigen queue",
        description="Run turbigen config files on parallel local workers.",
    )
    p.add_argument(
        "--queue-file",
        default=QUEUE_FILE_DEFAULT,
        help="Path to the queue file (one YAML path per line).",
    )
    p.add_argument(
        "--workers", type=int, default=4, help="Maximum number of parallel workers."
    )
    p.add_argument(
        "--purge",
        action="store_true",
        help="Cancel-all and exit: signal a running daemon (SIGHUP) "
        "or, if none is running, truncate the queue file directly.",
    )
    p.add_argument(
        "--print-unit",
        action="store_true",
        help="Print a systemd user unit file to stdout and exit.",
    )
    p.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Tail the daemon's systemd journal "
        "(execs `journalctl --user -u turbigen-queue -f`).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging (reserved; currently unused).",
    )
    return p


def cmd_queue(args):
    queue_file = Path(args.queue_file).expanduser()

    if args.follow:
        import shutil

        journalctl = shutil.which("journalctl")
        if journalctl is None:
            sys.stderr.write("journalctl not found on PATH.\n")
            sys.exit(1)
        os.execv(journalctl, [journalctl, "--user", "-u", "turbigen-queue", "-f"])

    if args.print_unit:
        import shutil

        uv = shutil.which("uv") or "uv"
        sys.stdout.write(
            UNIT_TEMPLATE.format(
                workdir=Path.cwd(),
                queue_file=queue_file,
                uv=uv,
            )
        )
        sys.stderr.write(
            "\n# Suggested install:\n"
            "#   mkdir -p ~/.config/systemd/user\n"
            "#   turbigen queue --print-unit > "
            "~/.config/systemd/user/turbigen-queue.service\n"
            "#   systemctl --user daemon-reload\n"
            "#   systemctl --user enable --now turbigen-queue\n"
        )
        return

    if args.purge:
        if signal_daemon():
            print("Sent SIGHUP to queue daemon (cancel-all).", flush=True)
        else:
            if queue_file.exists():
                queue_file.write_text("")
                print(f"Truncated {queue_file} (no daemon running).", flush=True)
            else:
                print(
                    "No daemon running and no queue file; nothing to purge.", flush=True
                )
        return

    if not queue_file.exists():
        queue_file.parent.mkdir(parents=True, exist_ok=True)
        queue_file.touch()
        print(f"Created queue file at {queue_file}", flush=True)

    serve(queue_file, args.workers)
