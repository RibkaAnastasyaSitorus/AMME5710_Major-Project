"""Compatibility launcher for the PPE video inference pipeline."""

# Reuse the main entry point so inference configuration stays in one place.
from main import main


if __name__ == "__main__":
    # Allow this file to remain an alternate command-line entry point.
    main()
