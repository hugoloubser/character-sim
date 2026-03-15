"""Robust path management utilities for the character creator system.

This module provides utilities for handling file paths and working directories
in a consistent, platform-independent way, ensuring scripts work correctly
regardless of where they're executed from.
"""

import os
import sys

from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory.

    Traverses up from the current file location to find the directory
    containing pyproject.toml, indicating the project root.

    Returns:
        Path to the project root directory.

    Raises:
        RuntimeError: If project root cannot be found.

    """
    # Start from the current file's location
    current = Path(__file__).resolve().parent

    # Traverse up to find pyproject.toml
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent

    msg = "Could not find project root (pyproject.toml not found)"
    raise RuntimeError(msg)


def get_src_directory() -> Path:
    """Get the src directory containing the character_creator module.

    Returns:
        Path to the src directory.

    """
    root = get_project_root()
    src_dir = root / "src"
    if not src_dir.exists():
        msg = f"src directory not found at {src_dir}"
        raise RuntimeError(msg)
    return src_dir


def ensure_src_in_path() -> None:
    """Ensure the src directory is in Python's module search path.

    This function adds the src directory to sys.path if it's not already there,
    allowing imports like `from character_creator import ...` to work correctly
    regardless of the current working directory.

    This is particularly important for:
    - Running scripts from different directories
    - Streamlit apps
    - Test runners with different working directories
    """
    src_path = str(get_src_directory())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def get_demos_directory() -> Path:
    """Get the demos directory.

    Returns:
        Path to the demos directory.

    """
    root = get_project_root()
    demos_dir = root / "demos"
    if not demos_dir.exists():
        msg = f"demos directory not found at {demos_dir}"
        raise RuntimeError(msg)
    return demos_dir


def get_config_file() -> Path:
    """Get the .env configuration file path.

    Returns:
        Path to the .env file in the project root.

    """
    return get_project_root() / ".env"


def set_working_directory(directory: Path | str) -> None:
    """Set the working directory for script execution.

    Args:
        directory: Path to the directory to change to.

    Raises:
        RuntimeError: If the directory does not exist.

    """
    path = Path(directory)
    if not path.exists():
        msg = f"Directory does not exist: {path}"
        raise RuntimeError(msg)
    os.chdir(path)


def setup_script_environment() -> None:
    """Set up the complete environment for running scripts.

    This function:
    1. Ensures the src directory is in Python's path
    2. Sets the working directory to the project root

    Call this at the beginning of any script that needs to work
    regardless of where it's executed from.
    """
    ensure_src_in_path()
    root = get_project_root()
    set_working_directory(root)
