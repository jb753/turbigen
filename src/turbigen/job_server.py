import concurrent.futures
import subprocess
import argparse
import time
import os
from pathlib import Path
import threading

QUEUE_FILE = "~/.turbigen/queue.txt"


def run_command(yaml_path):
    """Execute turbigen with the given YAML config file path"""
    # Use the no-job flag so that the jobs do not multiply
    command = ["turbigen", "--no-job", yaml_path]
    tid = str(threading.get_ident())[-3:]
    wstr = f"W{tid}"
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode == 0:
        print(f"{wstr} --- DONE: {yaml_path}")
    else:
        print(f"{wstr} --- FAIL: {yaml_path}")


def unqueue_job(queue_file, job):
    """Read a list of jobs and write it out again without the specified job"""

    with queue_file.open("r") as f:
        lines = f.readlines()

    with queue_file.open("w") as f:
        for line in lines:
            if not line.strip() == job:
                f.write(line)


def load_queue(queue_file):
    """Load the queue file and return a list of jobs"""
    with queue_file.open("r") as f:
        queue_jobs = [line.strip() for line in f]
        queue_jobs = [j for j in queue_jobs if j]
    return queue_jobs


def monitor_queue(queue_file, max_workers, poll_interval=5):
    """Monitor queue_file for new jobs and run them in parallel workers"""

    running = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            try:
                # Loop over jobs in the queue
                queue_jobs = load_queue(queue_file)
                for job in queue_jobs:
                    # Skip if another thread has already started this job
                    if job in running.values():
                        continue

                    # Start the job and add to the running dict
                    future = executor.submit(run_command, job)
                    running[future] = job

                # Delete completed jobs
                done = [f for f in running if f.done()]
                for f in done:
                    unqueue_job(queue_file, running[f])
                    del running[f]

                # Wait before checking the queue again
                time.sleep(poll_interval)

            except KeyboardInterrupt:
                print("Stopping monitoring due to keyboard interrupt.")
                break


def main():
    parser = argparse.ArgumentParser(
        description="Run turbigen config files on parallel workers"
    )
    parser.add_argument(
        "--queue_file",
        default=QUEUE_FILE,
        help="Path to the queue file with one YAML config path per line",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Clear the queue file before starting",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Polling interval in seconds for queue file",
    )
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"

    queue_file = Path(args.queue_file).expanduser()
    if not queue_file.exists():
        queue_file.parent.mkdir(parents=True, exist_ok=True)
        queue_file.touch()
        print(f"Created queue file at {queue_file}")
    elif args.purge:
        print(f"Purging queue file at {queue_file}")
        with queue_file.open("w") as f:
            f.write("")
    print(f"Monitoring {args.queue_file} for jobs.")
    monitor_queue(queue_file, args.workers, args.poll_interval)


if __name__ == "__main__":
    main()
