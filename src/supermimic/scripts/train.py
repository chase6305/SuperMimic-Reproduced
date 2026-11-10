"""Compatibility entry point for the guarded standalone training workflow."""

from supermimic.cli import train_main

if __name__ == "__main__":
    raise SystemExit(train_main())
