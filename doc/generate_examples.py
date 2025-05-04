"""Loop over examples, run them, and save the results as rst."""

import os
import re
import shutil
import subprocess
from pathlib import Path
import yaml

INPUT_DIR = "examples"
OUTPUT_DIR = "doc/examples"


# Function to process a single example
def run_example(input_yaml):
    # Run turbigen as a subprocess and capture the log on stderr
    print(f"Running example: {input_yaml}...")
    out = subprocess.run(
        ["turbigen", str(input_yaml)],
        stderr=subprocess.PIPE,
        text=True,
    )

    # Load the yaml
    with open(input_yaml, "r") as f:
        yaml_str = f.read()
    yaml_indented = "\n".join(
        "   " + line if line.strip() != "" else "" for line in yaml_str.splitlines()
    )
    # Indent the output
    log_indented = "\n".join(["    " + l for l in out.stderr.splitlines()])

    # Read comment lines from start of the yaml
    with open(input_yaml, "r") as f:
        lines = f.readlines()
        comments = [line for line in lines if line.startswith("#")]

    # Attempt to parse the title
    try:
        title = comments[0].split("#")[-1].strip()
    except Exception:
        title = "Example"

    # Parse workdir from the log
    regex = r"Working directory:\s*([^\s]+)"
    workdir = Path(re.search(regex, out.stderr).group(1))

    # Convert pdf to svg
    post_pdf = workdir / "post.pdf"
    post_svg = workdir / f"{input_yaml.name}_post_%d.svg"
    print(f"Converting {post_pdf} to {post_svg}")
    subprocess.run(["pdf2svg", str(post_pdf), str(post_svg), "all"])

    # List the images in the workdir
    images = sorted(workdir.glob("*_post_*.svg"))

    # Copy the images to the output directory
    for image in images:
        # Copy the image to the output directory
        shutil.copy(image, OUTPUT_DIR)

    # Assemble the images into a single string
    images_str = "\n".join(
        [f".. image:: {image.name}\n   :width: 100%\n" for image in images]
    )

    # Now assemble an rst file
    rst = f"""{"=" * len(title)}
{title}
{"=" * len(title)}

:download:`Download this example <../../{input_yaml}>`

Input file
==========


.. code-block:: yaml

{yaml_indented}

Log output
==========

.. code-block:: none

{log_indented}

Plots
=====

{images_str}

"""

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write the rst file to the output directory
    output_file = Path(OUTPUT_DIR) / (input_yaml.stem + ".rst")
    with open(output_file, "w") as f:
        f.write(rst)

    print(f"Example saved to: {output_file}")


if __name__ == "__main__":
    # Open hashes file if it exists
    hashes_file = Path(OUTPUT_DIR) / "hashes.yaml"
    if hashes_file.exists():
        with hashes_file.open("r") as f:
            hashes = yaml.safe_load(f)
    else:
        hashes = {}

    # Loop over  all examples
    print("Running examples")
    for example in Path(INPUT_DIR).iterdir():
        if example.suffix == ".yaml":
            # Take md5 hash of the yaml file
            md5 = subprocess.run(
                ["md5sum", str(example)],
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.split()[0]

            # Check if the hash is already in the hashes file
            print(hashes.get(example.stem))
            print(md5)
            if hashes.get(example.stem) == md5:
                print(f"Skipping {example} (already run)")
                continue
            else:
                hashes[example.stem] = md5

            # Run the example
            run_example(example)

    # Save the hashes
    with hashes_file.open("w") as f:
        yaml.dump(hashes, f)
