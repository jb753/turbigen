"""Start the AutoGrid server shell script with args."""

import os
import argparse
import subprocess


def _make_argparser():
    """Set up argument parsing"""

    parser = argparse.ArgumentParser(
        description=(
            "This AutoGrid server automates mesh generation during a "
            "turbigen run on a remote machine. It monitors a QUEUE_FILE "
            "for temporary directories to process, which are created "
            "and queued by the remote machine. The server uses AutoGrid "
            "batch scripts to create a computational mesh and touches an "
            "extra file to notify the remote machine of sucess. If the "
            "delete flag is specified, it then removes the tempory dir."
        ),
        usage="%(prog)s [--delete] QUEUE_FILE",
        add_help="False",
    )

    parser.add_argument(
        "QUEUE_FILE",
        help=(
            "filename of the queue of directories, "
            "if omitted default $HOME/.ag_queue.txt"
        ),
        nargs="?",
        default="$HOME/.ag_queue.txt",
    )

    parser.add_argument(
        "-d",
        "--delete",
        help="delete temporary files on completion",
        action="store_true",
    )

    return parser


def main():
    # Get file path to the shell script
    script_name = os.path.join(os.path.dirname(__file__), "ag_server.sh")

    # Add args
    args = _make_argparser().parse_args()
    cmd_str = [script_name,] + [
        os.path.expandvars(args.QUEUE_FILE),
    ]
    if args.delete:
        cmd_str += ["--delete"]

    # Run the command
    subprocess.run(cmd_str)
