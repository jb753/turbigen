"""Start the AutoGrid server shell script."""
import os
import subprocess


def main():

    # Get file path to the shell script
    script_name = os.path.join(os.path.dirname(__file__), "ag_server.sh")
    subprocess.run(script_name)
