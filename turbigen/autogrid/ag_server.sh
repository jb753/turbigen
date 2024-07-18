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

echo "AG server starting, waiting for jobs in $WAT_FILE"

if [ "$DELETE" ]; then
    echo "Deletion of completed meshes is enabled"
else
    echo "Will not delete completed meshes"
fi

# Wait for a random amount of time to avoid races when
# multiple servers start at the same time
sleep $((RANDOM % 20))

# Make sure the queue file exists
touch "$WAT_FILE"

# Loop forever
while :
do

    # Just sleep if no jobs waiting
    SCR=$(head -1 "$WAT_FILE")

    if [ -z "$SCR" ]; then
        sleep 25
        sleep $((RANDOM % 10))

    # If jobs are waiting, check for license
    elif "$IGGBIN" -autogrid5 -print -batch 2>&1 | grep "All licenses are in use" &> /dev/null; then

        echo "$(date -Iminutes): no free license, $(wc -l $WAT_FILE) jobs waiting."

        sleep 115
        sleep $((RANDOM % 10))

    # If jobs are waiting and we have a license, do the meshing
    else

        TEMP_SED=$(sed "\@$SCR@d" "$WAT_FILE")
        echo "$TEMP_SED" | sed -e '/^$/d' > "$WAT_FILE"

        echo "$(date -Iminutes): $SCR meshing."

        cd "$SCR"

        "$IGGBIN" -autogrid5 -batch -print -script script_ag.py2 &> out.log \
            && "$IGGBIN" -batch -script script_igg.py2 &> out_2.log# && chmod a+r mesh.g mesh.bcs

        if [ $(find . -name "*.g") ]; then
            echo "$(date -Iminutes): $SCR completed."
            touch finished
            cd ..
            sleep 30
            # As a safety check, only delete if 'tmp' is in the name
            if [ "$DELETE" ]; then
                if [[ $SCR =~ "tmp" ]]; then
                    rm -rf "$SCR"
                fi
            fi
        else
            touch failed
            echo "$(date -Iminutes): $SCR failed."
            cd ..
        fi


    fi

done
