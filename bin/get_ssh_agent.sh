#!/bin/bash
# Set env vars for ssh agent, picking up an existing one if already running

TURBIGEN_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPTPATH="$TURBIGEN_ROOT/get_ssh_agent.sh"
if [ -n "$1" ]; then
    # If a host is specified then run on that host
    `ssh -o BatchMode=yes "$1" "$SCRIPTPATH"`
else

    # If AGENT PID is unset, check that process is still running
    [ -z "$SSH_AGENT_PID" ] &&
        export SSH_AGENT_PID=$(ps x | grep -vE 'defunct|grep' | grep -w ssh-agent | tail -1 | awk '{print $1}')

    # Start a new agent if we don't have one
    if [ -z "$SSH_AGENT_PID" ]; then
        pkill ssh-agent 2> /dev/null
        eval $(ssh-agent) && ssh-add
    fi

    # Wait for filesystem
    sleep 1

    # Get pid from the process listing
    export SSH_AGENT_PID=$(ps x | grep -vE 'defunct|grep' | grep -w ssh-agent | tail -1 | awk '{print $1}')

    # Get socket from the file system
    export SSH_AUTH_SOCK=$(ls /tmp/ssh-*/agent.${SSH_AGENT_PID:0:5}*)

fi

echo "export SSH_AGENT_PID=$SSH_AGENT_PID"
echo "export SSH_AUTH_SOCK=$SSH_AUTH_SOCK"
