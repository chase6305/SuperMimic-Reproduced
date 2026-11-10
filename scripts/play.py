"""Compatibility shortcut to the guarded policy playback CLI."""

import sys

from supermimic.cli import main

if __name__ == "__main__":
    arguments = [
        "--checkpoint" if value == "--ckpt" else "--motions" if value == "--npz-dir" else value
        for value in sys.argv[1:]
    ]
    raise SystemExit(main(["play", *arguments]))
