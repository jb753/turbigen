#!/bin/bash
# Run AutoGrid jobs as they appear in a queue
# Usage: ag_server.sh QUEUE_FILE

# Set path to IGG binary
# IGGBIN="/opt/numeca/bin/igg"
IGGBIN="igg"

# AutoGrid needs to find the screen
export DISPLAY=:0.0

# Where are the temporary mesh files stored
mkdir -p $HOME/tmp

# Get arguments
WAT_FILE="$1"
DELETE="$2"
WORKER="$3"
WORKER="W$WORKER"

# Function to take first line from a file avoiding race conditions
pop_first_line() {
    local queue_file="$1"
    local temp_file="${queue_file}.tmp"
    local job=""

    # Ensure the queue file exists
    [ -f "$queue_file" ] || return 1

    # Use flock to ensure exclusive access to the queue file
    {
        flock -x 200

        # Read the first line from the queue file
        job=$(head -n 1 "$queue_file")

        # If the queue file is not empty
        if [ -n "$job" ]; then

            # Remove the first line and save the rest to a temporary file
            tail -n +2 "$queue_file" > "$temp_file"

            # Move the temporary file back to the queue file
            mv -f "$temp_file" "$queue_file"

            # Restore access permissions to all users
            chmod a+rw "$queue_file"

        fi

    } 200<"$queue_file"

    # Return the job that was popped
    echo "$job"
}

# Function to add a job at first line
push_first_line() {
    local queue_file="$1"
    local job="$2"
    local temp_file="${queue_file}.tmp"

    # Ensure the queue file exists; create it if it doesn't
    [ -f "$queue_file" ] || touch "$queue_file"
    chmod a+rw "$queue_file"

    # Use flock to ensure exclusive access to the queue file
    {
        flock -x 200

        # Create a temporary file with the new job as the first line
        echo "$job" > "$temp_file"

        # Append the existing queue to the temporary file
        cat "$queue_file" >> "$temp_file"

        # Move the temporary file back to the queue file
        mv -f "$temp_file" "$queue_file"

        # Restore access permissions to all users
        chmod a+rw "$queue_file"

    } 200<"$queue_file"
}


check_license_avail() {
    local command="$IGGBIN -autogrid5 -print -batch"
    local pattern="License call failed for feature FIDELITY_AUTOMESH"

    # Execute the command and capture its output
    local output
    output=$(eval "$command" 2>&1)

    # Check if the pattern is found in the output
    if echo "$output" | grep -q "$pattern"; then
        return 1  # Pattern found, so license is NOT free
    else
        return 0  # Pattern not found, we have a free license
    fi
}

if [ "$DELETE" -ne 0 ]; then
    DEL_STR="deleting completed meshes."
else
    DEL_STR="not deleting meshes."
fi

echo "$WORKER --- Starting, queue file $WAT_FILE, $DEL_STR"

# Make sure the queue file exists
touch "$WAT_FILE"
chmod a+rw "$WAT_FILE"

# Loop forever
while :
do

    # Read first waiting job in the queue
    SCR=$(pop_first_line "$WAT_FILE")
    # SCR=$(head -1 "$WAT_FILE")

    # Just sleep if no jobs waiting
    if [ -z "$SCR" ]; then
        sleep $((RANDOM % 10))

    # If by some malfunction the queued directory does not exist, skip it
    elif [ ! -d "$SCR" ]; then
        echo "$WORKER --- $(date -Iminutes): $SCR queued but does not exist."

    # If by some malfunction the queued directory has already began meshing, skip it
    elif [ -f "$SCR/out.log" ]; then
        echo "$WORKER --- $(date -Iminutes): $SCR queued but already meshing."

    # If jobs are waiting and we have a license, do the meshing
    elif check_license_avail; then

        # TEMP_SED=$(sed "\@$SCR@d" "$WAT_FILE")
        # echo "$TEMP_SED" | sed -e '/^$/d' > "$WAT_FILE"

        echo "$WORKER --- $(date -Iminutes): $SCR meshing."

        cd "$SCR"

        "$IGGBIN" -autogrid5 -batch -print -script script_ag.py2 &> out.log \
            && "$IGGBIN" -batch -script script_igg.py2 &> out_2.log

        if [ $(find . -name "*.g") ]; then
            echo "$WORKER --- $(date -Iminutes): $SCR completed."
            chmod a+rw mesh.{g,bcs}
            touch finished
            cd ..
            sleep 20
            # As a safety check, only delete if 'tmp' is in the name
            if [ "$DELETE" -ne 0 ]; then
                if [[ $SCR =~ "tmp" ]]; then
                    rm -rf "$SCR"
                fi
            fi
        else
            touch failed
            echo "$WORKER --- $(date -Iminutes): $SCR failed."
            cd ..
        fi

    # If jobs are waiting but we do not have license
    else

        echo "$WORKER --- $(date -Iminutes): no free license, $(wc -l $WAT_FILE) jobs waiting."

        # We can't mesh it right now so put the job back in the queue
        push_first_line "$WAT_FILE" "$SCR"

        sleep 120

    fi

done
