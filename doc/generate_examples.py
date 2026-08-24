"""Run the examples, so that the documentation can show what they produce.

One job, and the rest of the pipeline is arranged so that it stays one job.
The runs write into a build directory of their own, named after the example, so
this never touches the source tree; the Sphinx directive reads that directory;
and nothing here templates rst, hashes a file, converts an image or parses a
log. The predecessor did all four, and each was a coupling that broke: it
scraped the working directory out of the log, so a change to one INFO line
stopped it; it cached on an md5 written to a gitignored file, so the cache
never survived a clone; and it pasted the log into a generated rst, so the
documentation could disagree with the run that produced it.

Examples are independent, so they run at once. Ember is one process per run
with no thread pool of its own, which makes one core per example the right
model and a thread per subprocess enough to supervise them.

Usage::

    make generate-examples                         # what is out of date
    uv run python doc/generate_examples.py -j 2    # two at a time
    uv run python doc/generate_examples.py -f      # everything again
    uv run python doc/generate_examples.py fan     # just this one
"""

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "examples"
BUILD_DIR = ROOT / "doc" / "_examples"

OUTPUT_NAME = "output.yaml"
LOG_NAME = "log_turbigen.txt"

LOG_TAIL = 20
"""Lines of a failed run's log to print, being the ones that say why."""


def build_dir(example):
    """Return the directory `example` is run in."""
    return BUILD_DIR / example.stem


def is_stale(example):
    """Return whether `example` has changed since it was last run.

    Modification times against the output the run wrote, which is all the
    ordering that is needed and is what every other build tool uses. An
    interrupted run leaves no `output.yaml`, so it counts as stale and is
    picked up next time rather than being taken for finished.
    """
    output = build_dir(example) / OUTPUT_NAME
    if not output.exists():
        return True

    return example.stat().st_mtime > output.stat().st_mtime


def run(example):
    """Run one example, returning its exit status.

    `--svg` because the documentation places the figures one at a time, and
    `-o` because a build product does not belong in `examples/` --- which is
    also what makes the output path known here, rather than something to ask
    the run about afterwards. No `--force` is needed: the directory is emptied
    below, so there is never an answer in it to refuse to replace.
    """
    out_dir = build_dir(example)
    print(f"Running {example.name} in {out_dir.relative_to(ROOT)}")

    # Emptied first, because turbigen appends to `log_turbigen.txt` rather than
    # replacing it: a directory run twice holds both transcripts, and the
    # documentation would show a reader the run that failed above the one that
    # worked. Everything in here is a build product, so there is nothing to
    # lose by starting from nothing.
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Output goes to the run's own log file, which turbigen writes into the
    # build directory. Capturing it here as well would only interleave several
    # runs into one unreadable stream.
    return subprocess.run(
        [
            "turbigen",
            "run",
            "--svg",
            "-o",
            str(out_dir),
            str(example),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode


def report_failure(example, status):
    """Print why one example failed, from the log the run left behind."""
    print(f"\n{example.name} exited {status}", file=sys.stderr)

    log = build_dir(example) / LOG_NAME
    if not log.exists():
        print(f"  no {LOG_NAME} was written", file=sys.stderr)
        return

    print(f"  last {LOG_TAIL} lines of {log.relative_to(ROOT)}:", file=sys.stderr)
    for line in log.read_text().splitlines()[-LOG_TAIL:]:
        print(f"  | {line}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help="examples to run at once; the default is one per core",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="run every example, rather than only those that have changed",
    )
    parser.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help="examples to run, named without the .yaml; the default is all",
    )
    args = parser.parse_args(argv)

    examples = sorted(INPUT_DIR.glob("*.yaml"))
    if not examples:
        print(f"No examples in {INPUT_DIR}", file=sys.stderr)
        return 1

    # Naming examples is what lets a caller pay for the ones it wants. CI runs
    # a single cheap case; the documentation build runs the lot. An unknown
    # name is an error rather than an empty selection, so a typo in a workflow
    # cannot quietly turn the job into a no-op that passes.
    if args.names:
        by_name = {e.stem: e for e in examples}
        unknown = [n for n in args.names if n not in by_name]
        if unknown:
            print(
                f"No such example(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(by_name))}",
                file=sys.stderr,
            )
            return 1
        examples = [by_name[n] for n in dict.fromkeys(args.names)]

    stale = examples if args.force else [e for e in examples if is_stale(e)]
    for example in examples:
        if example not in stale:
            print(f"Skipping {example.name}, already run")

    if not stale:
        return 0

    # No more workers than there is work, so a single example does not open a
    # pool the width of the machine.
    jobs = args.jobs or min(len(stale), os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        statuses = list(pool.map(run, stale))

    # Every failure, not the first: with runs happening at once, stopping at
    # one would hide whatever the others had to say and leave which of them
    # failed depending on the order they happened to finish in.
    failed = [(e, s) for e, s in zip(stale, statuses) if s]
    for example, status in failed:
        report_failure(example, status)

    if failed:
        names = ", ".join(e.name for e, _ in failed)
        print(f"\n{len(failed)} of {len(stale)} examples failed: {names}")
        return 1

    print(f"\nRan {len(stale)} example(s) into {BUILD_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
