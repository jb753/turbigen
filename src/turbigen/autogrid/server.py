"""Start the AutoGrid server shell script with args."""

import os
import argparse
import subprocess


def _make_argparser():
    """Set up argument parsing"""

    parser = argparse.ArgumentParser(
        description=(
            "This AutoGrid server automates mesh generation during a "
            "turbigen run on a remote machine. It monitors a queue file "
            "inside a WORKDIR for temporary directories to process, which "
            "are created "
            "and queued by the remote machine. The server uses AutoGrid "
            "batch scripts to create a computational mesh and touches an "
            "extra file to notify the remote machine of sucess. If the "
            "delete flag is specified, it then removes the tempory dir."
        ),
        usage="%(prog)s [--delete] [--workers=1] WORKDIR",
        add_help="False",
    )

    parser.add_argument(
        "WORKDIR",
        help=("path to working directory storing queue and completed meshes. "),
    )

    parser.add_argument(
        "-d",
        "--delete",
        help="delete temporary files on completion",
        action="store_true",
    )

    parser.add_argument(
        "-w",
        "--workers",
        help="number of parallel workers, default serial",
        default=1,
        type=int,
    )

    return parser


def main():
    # Get file path to the shell script
    script_name = os.path.join(os.path.dirname(__file__), "ag_server.sh")

    # Add args
    args = _make_argparser().parse_args()
    queue_file = os.path.join(args.WORKDIR, "queue.txt")
    cmd_str = [script_name,] + [
        os.path.expandvars(queue_file),
    ]

    if args.delete:
        cmd_str += ["1"]
    else:
        cmd_str += ["0"]

    # Start the workers
    worker_procs = []
    for i in range(args.workers):
        worker_procs.append(subprocess.Popen(cmd_str + [str(i)]))

    # Wait for all subprocesses to complete
    for proc in worker_procs:
        proc.wait()
