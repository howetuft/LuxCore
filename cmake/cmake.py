# SPDX-FileCopyrightText: 2025 Howetuft
#
# SPDX-License-Identifier: Apache-2.0

"""This script wraps cmake calls for LuxCore build.

It is intended to get the same behaviour between Windows and *nix os
"""

import os
import subprocess
import sys
import logging
import shutil
import argparse
from pathlib import Path
from functools import cache
import re
from enum import Enum

import make_deps

# External variables
BINARY_DIR = Path(os.getenv("LUX_BINARY_DIR", "out"))
SOURCE_DIR = Path(os.getenv("LUX_SOURCE_DIR", os.getcwd()))
BUILD_TYPE = os.getenv("LUX_BUILD_TYPE", "Release")

# Computed variables
BUILD_DIR = BINARY_DIR / "build"
INSTALL_DIR = BINARY_DIR / "install" / BUILD_TYPE

# Logger
logger = logging.getLogger("LuxCore Build")


# Preset stuff
class PresetType(Enum):
    """CMake preset types."""

    CONFIG = "configure"
    BUILD = "build"
    TEST = "test"
    PACKAGE = "package"


PRESET_PATTERN = re.compile(r"\s*\"([a-zA-Z\-]+)\".*")


@cache
def ensure_cmake_app():
    """Ensure cmake is installed."""
    logger.debug("Looking for cmake")
    if not (res := shutil.which("cmake")):
        logger.error("CMake not found!")
        sys.exit(1)
    logger.debug("CMake found: '%s'", res)
    return res


def run_cmake(args, **kwargs):
    """Run cmake statement."""
    cmake_app = ensure_cmake_app()
    args = [cmake_app] + args
    logger.debug(args)
    res = subprocess.run(args, shell=False, check=False, **kwargs)
    if res.returncode:
        logger.error("Error while executing cmake")
        print(res.stdout)
        print(res.stderr)
        sys.exit(1)
    return res


def config(_):
    """CMake config."""
    cmd = [
        "--preset conan-default",
        f"-DCMAKE_INSTALL_PREFIX={str(INSTALL_DIR)}",
        f"-S {str(SOURCE_DIR)}",
    ]
    run_cmake(cmd)


PRESETS = {
    "Release": "conan-release",
    "Debug": "conan-debug",
    "RelWithDebInfo": "conan-relwithdebinfo",
    "MinSizeRel": "conan-minsizerel",
}


def get_preset_from_build_type(build_type):
    """Get conan preset from build type."""
    try:
        preset = PRESETS[build_type]
    except KeyError:
        logger.error("Unknown build type '%s'", build_type)
        logger.error("Valid values (case sensitive) are: %s", PRESETS.keys())
        sys.exit(1)
    if preset not in (presets := get_presets(PresetType.BUILD)):
        logger.error("Preset '%s' missing", preset)
        logger.error("Available presets: %s", presets)
    return preset


def build(args):
    """CMake build."""
    preset = get_preset_from_build_type(BUILD_TYPE)
    cmd = [
        "--build",
        f"--preset {preset}",
        f"--target {args.target}",
    ]
    run_cmake(cmd)


def install(args):
    """CMake install."""
    cmd = [
        "--install",
        str(BUILD_DIR),
        f"--prefix {INSTALL_DIR}",
        f"--config {BUILD_TYPE}",
        f"--component {args.target}",
    ]
    run_cmake(cmd)


def build_and_install(args):
    """CMake build and install."""
    build(args)
    install(args)


def get_presets(preset_type: PresetType):
    """CMake get presets for a given type."""
    preset_type = str(preset_type)
    cmd = [f"--list-presets={preset_type}"]
    res = run_cmake(cmd, capture_output=True, text=True)
    presets = [
        preset[1]
        for line in res.stdout.splitlines()
        if (preset := PRESET_PATTERN.fullmatch(line)) is not None
    ]
    return presets


def get_all_presets():
    """CMake get all presets."""
    presets = {
        preset_type.value: get_presets(preset_type)
        for preset_type in PresetType.__members__.values()
    }
    return presets


def list_presets(_):
    """List all presets."""
    print(get_all_presets())


def clean(_):
    """CMake clean."""
    for preset in get_presets(PresetType.BUILD):
        logger.info("Cleaning preset '%s'", preset)
        cmd = [
            "--build",
            f"--preset {preset}",
            "--target clean",
        ]
        run_cmake(cmd)


def clear(_):
    """Clear binary directory."""
    # We just remove the subdirectories, in order to avoid
    # unwanted removals if BINARY_DIR points to a wrong directory
    for subdir in ("build", "dependencies", "install"):
        directory = BINARY_DIR / subdir
        logger.info("Removing '%s'", directory)
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            logger.debug("'%s' not found", directory)


def deps(_):
    """Install dependencies."""
    make_deps.main([f"--output={BINARY_DIR}"])


def main():
    """Entry point."""
    # Set-up logger
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

    # Get command-line parameters
    parser = argparse.ArgumentParser(prog="make", add_help=False)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers()

    # Dependencies
    parser_deps = subparsers.add_parser("deps")
    parser_deps.set_defaults(func=deps)

    # List Presets
    parser_presets = subparsers.add_parser("list-presets")
    parser_presets.set_defaults(func=list_presets)

    # Config
    parser_config = subparsers.add_parser("config")
    parser_config.set_defaults(func=config)

    # Build
    parser_build_and_install = subparsers.add_parser("build-and-install")
    parser_build_and_install.add_argument("target")
    parser_build_and_install.set_defaults(func=build_and_install)

    # Install
    parser_install = subparsers.add_parser("install")
    parser_install.add_argument("target")
    parser_install.set_defaults(func=install)

    # Clean
    parser_clean = subparsers.add_parser("clean")
    parser_clean.set_defaults(func=clean)

    # Clear
    parser_clear = subparsers.add_parser("clear")
    parser_clear.set_defaults(func=clear)

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode")
    args.func(args)


if __name__ == "__main__":
    main()
