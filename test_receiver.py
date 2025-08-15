#!/usr/bin/env python3
"""
Test script to receive a piped grid structure from the external solver interface.
This script receives the grid via named pipe, processes it, and sends it back via another named pipe.
"""

import sys
import pickle


def main():
    """Receive grid data from input pipe and send processed grid back via output pipe"""

    print("RECIEVER START")

    if len(sys.argv) != 3:
        print("Usage: test_receiver.py <input_pipe> <output_pipe>")
        sys.exit(1)

    input_pipe = sys.argv[1]
    output_pipe = sys.argv[2]

    print(f"Reading from pipe: {input_pipe}")

    # Read the pickled grid data from input pipe
    with open(input_pipe, "rb") as f:
        grid = pickle.load(f)

    print(f"Received grid with {len(grid)} blocks")

    # CFD solver would go here
    # For now, just return the grid unchanged
    print("CFD solver processing...")

    print(f"Writing to pipe: {output_pipe}")

    # Send the processed grid back via output pipe
    with open(output_pipe, "wb") as f:
        pickle.dump(grid, f)

    print("RECIEVER END")


if __name__ == "__main__":
    main()
