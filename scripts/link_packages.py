import os
import sys
import site
import pathlib
import shutil


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up a symbolic link for a Python package (editable install style)."
    )
    parser.add_argument(
        "source_dir", help="Path to the source directory of the package"
    )
    parser.add_argument("package_name", help="Name of the package to link")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without making changes"
    )

    args = parser.parse_args()

    src_dir = pathlib.Path(args.source_dir).resolve()
    package_name = args.package_name
    dry_run = args.dry_run

    if not src_dir.exists():
        print(f"Error: source directory {src_dir} does not exist.")
        sys.exit(1)

    # Find the site-packages directory
    try:
        site_packages = next(p for p in site.getsitepackages() if "site-packages" in p)
    except StopIteration:
        print("Could not locate site-packages directory.")
        sys.exit(1)

    link_path = pathlib.Path(site_packages) / package_name

    if link_path.exists() or link_path.is_symlink():
        print(
            f"Would remove existing: {link_path}"
            if dry_run
            else f"Removing existing: {link_path}"
        )
        if not dry_run:
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink()
            elif link_path.is_dir():
                shutil.rmtree(link_path)

    print(
        f"Would create symlink: {link_path} -> {src_dir}"
        if dry_run
        else f"Creating symlink: {link_path} -> {src_dir}"
    )
    if not dry_run:
        try:
            os.symlink(src_dir, link_path, target_is_directory=True)
        except OSError as e:
            print(f"Failed to create symlink: {e}")
            sys.exit(1)

    print(f"Editable link {'would be' if dry_run else 'was'} created at: {link_path}")


if __name__ == "__main__":
    main()
