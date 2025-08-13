"""SSH connection management."""

import subprocess
import os
from time import sleep


class SSHConnection:
    def __init__(self, remote_host, via_host=None):
        self.remote_host = remote_host
        self.via_host = via_host
        self.base_ssh = [
            "ssh",
            "-o",
            "BatchMode=yes",
            # "-o",
            # "ControlMaster=no",
            # "-o",
            # "ControlPath=none",
            # "-o",
            # "ControlPersist=no",
        ]
        self.base_scp = ["scp", "-q"]
        if self.via_host:
            try:
                self.get_via_agent()
                self.run_remote("hostname", check=False, timeout=1)
            except Exception:
                print(f"Failed to find ssh-agent on via_host {self.via_host}.")
                print("Attempting to (re)start the agent.")
                print("This may require a password prompt.")
                self.start_via_agent()
                self.get_via_agent()
                # For some reason the first attempt to connect to the
                # remote host via the via_host fails, so we try again
                # after a short delay
                try:
                    self.run_remote("hostname", check=False, timeout=1)
                except Exception:
                    pass
                self.run_remote("hostname", check=True)
        else:
            self.via_msg = ""

    def _execute(self, command, check, capture=True, timeout=60):
        ssh_cmd = command

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=capture,
                text=capture,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            err_msg = f"ssh command timed out: {' '.join(ssh_cmd)}"
            raise Exception(err_msg)

        if check and result.returncode:
            raise Exception(
                "ssh connection failed.\n"
                f"COMMAND: {' '.join(ssh_cmd)}\n"
                f"RETURN CODE: {result.returncode}\n"
                "STDOUT:\n"
                f"{result.stdout.strip()}\n"
                "STDERR:\n"
                f"{result.stderr.strip()}"
            )

        return result

    def _via_prefix(self):
        if (
            not self.via_host
            or not os.environ.get("SSH_AUTH_SOCK")
            or not os.environ.get("SSH_AGENT_PID")
        ):
            return []
        env_str = [f"{k}={os.environ[k]}" for k in ["SSH_AUTH_SOCK", "SSH_AGENT_PID"]]
        return self.base_ssh + [self.via_host] + env_str

    def _scp(self, command, check):
        assert (
            "'" not in command
        ), f"Command should not contain single quotes: {command}"
        return self._execute(self._via_prefix() + self.base_scp + command, check)

    def start_via_agent(self):
        """Start the SSH agent on the via_host."""
        # Kill any existing ssh-agent processes on via_host
        self._execute(["ssh", "-t", self.via_host, "pkill ssh-agent"], check=False)
        sleep(3)
        self._execute(
            ["ssh", "-t", self.via_host, "eval $(ssh-agent) && ssh-add"],
            check=False,
            capture=False,
        )

    def run_via(self, command, check=True, timeout=60):
        """Run a command on the via_host directly."""
        if not self.via_host:
            raise ValueError("No via_host defined")
        return self._execute(
            self.base_ssh + [self.via_host] + [command], check, timeout=timeout
        )

    def run_remote(self, command, check=True, timeout=60):
        """Run a command on the remote host, optionally via via_host."""
        return self._execute(
            self._via_prefix() + self.base_ssh + [self.remote_host] + [command],
            check,
            timeout=timeout,
        )

    def get_via_agent(self):
        """Attempt to find SSH agent on via_host and get env vars."""

        if self.run_via("true", timeout=10).returncode:
            raise Exception(f"Via host {self.via_host} is not reachable.")

        # Find the ssh-agent process running on the via_host
        pid = self.run_via(
            r"""ps aux | grep $(whoami) |  tr -s " " | cut -d" " -f2,11 | grep ssh-agent | cut -d" " -f1""",
            check=False,
        ).stdout.strip()

        # Use the freshest pid if multiple are returned
        if "\n" in pid:
            pid = pid.split("\n")[-1].strip()

        if not pid:
            raise Exception(
                f"Could not locate ssh-agent process running on via host {self.via_host}."
            )

        # Find the SSH_AUTH_SOCK associated with the agent
        sock = self.run_via(f"ls /tmp/ssh-*/agent.{pid[:5]}*").stdout.strip()
        if not sock:
            raise Exception(
                f"Could not locate ssh-agent socket on via host {self.via_host}."
            )

        os.environ["SSH_AUTH_SOCK"] = sock
        os.environ["SSH_AGENT_PID"] = pid

    def copy_to_remote(self, local_path, remote_path):
        local_paths = os.path.abspath(local_path).split()
        command = [p for p in local_paths] + [f"{self.remote_host}:{remote_path}"]
        return self._scp(command, check=True)

    def copy_from_remote(self, remote_path, local_path):
        command = [
            f"{self.remote_host}:{remote_path}",
            f"{os.path.abspath(local_path)}",
        ]
        return self._scp(command, check=True)


if __name__ == "__main__":
    # Without via
    # ssh_conn = SSHConnection(remote_host="login-q-4")
    # assert ssh_conn.run_remote("hostname").stdout.strip() == ssh_conn.remote_host
    # ssh_conn.copy_to_remote("./testfile", "~/beans")
    # ssh_conn.copy_from_remote("~/beans", "./beans2")

    ssh_conn = SSHConnection(remote_host="gp-111", via_host="login-q-2")
    assert ssh_conn.run_remote("hostname").stdout.strip() == ssh_conn.remote_host
    # ssh_conn.start_agent()
    ssh_conn.copy_to_remote("./testfile", "~/beans")
    ssh_conn.copy_from_remote("~/beans", "./beans2")
    print(ssh_conn.run_via("hostname").stdout.strip())
    print(ssh_conn.run_remote("hostname").stdout.strip())
    # ssh_conn.copy_to_remote("./testfile", "~/beans")
