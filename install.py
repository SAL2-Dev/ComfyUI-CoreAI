"""
install.py — Post-install hook for comfy-cli / ComfyUI Manager.

Called automatically when the node is installed via:
    comfy node install ComfyUI-CoreAI

Verifies the environment (Apple Silicon, macOS version) and pre-downloads
the coreai-runner binary so the first predict() call is instant.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path

logger = logging.getLogger("ComfyUI-CoreAI")

# Minimum requirements
REQUIRED_ARCH = "arm64"
REQUIRED_OS = "Darwin"
REQUIRED_MACOS_MAJOR = 26  # Foundation Models requires macOS 26+


def check_environment() -> tuple[bool, str]:
    """
    Verify the host can run coreai-runner.
    Returns (ok, message).
    """
    arch = platform.machine()
    os_name = platform.system()

    if os_name != REQUIRED_OS:
        return False, (
            f"ComfyUI-CoreAI requires macOS. Detected: {os_name}. "
            "Core AI / Foundation Models framework is Apple-only."
        )

    if arch != REQUIRED_ARCH:
        return False, (
            f"ComfyUI-CoreAI requires Apple Silicon (arm64). Detected: {arch}. "
            "Intel Macs are not supported — Core AI requires Apple Silicon."
        )

    # Check macOS version
    macos_version = platform.mac_ver()[0]
    if macos_version:
        major = int(macos_version.split(".")[0])
        if major < REQUIRED_MACOS_MAJOR:
            return False, (
                f"ComfyUI-CoreAI requires macOS {REQUIRED_MACOS_MAJOR}+ "
                f"(Foundation Models framework). Detected: macOS {macos_version}."
            )

    return True, f"Environment OK: {arch} macOS {macos_version}"


def pre_download_binary() -> bool:
    """
    Pre-download the coreai-runner binary so the first inference call is instant.
    Returns True if binary is ready, False if it will be lazy-loaded instead.
    """
    pkg_dir = Path(__file__).parent / "comfyui_coreai" / "bin"
    binary = pkg_dir / "coreai-runner"

    if binary.exists() and os.access(binary, os.X_OK):
        logger.info("coreai-runner binary already present at %s", binary)
        return True

    ok, msg = check_environment()
    if not ok:
        logger.warning("Skipping binary download: %s", msg)
        return False

    try:
        from comfyui_coreai.bridge import _download_binary

        logger.info("Pre-downloading coreai-runner binary...")
        path = _download_binary()
        logger.info("Binary ready at %s", path)
        return True
    except Exception as e:
        logger.warning(
            "Could not pre-download binary (will try on first use): %s", e
        )
        return False


def on_install() -> None:
    """
    Entry point called by comfy-cli / ComfyUI Manager after install.
    """
    logger.info("=" * 60)
    logger.info("ComfyUI-CoreAI installation check")
    logger.info("=" * 60)

    ok, msg = check_environment()
    if ok:
        logger.info(msg)
    else:
        logger.error(msg)
        logger.error(
            "The node will be installed but will NOT function on this machine. "
            "Models will not be downloadable and inference will fail."
        )
        # Don't fail the install — the user might be installing on a
        # non-Apple machine by mistake, and we want the error message
        # to be visible in ComfyUI's node management UI.
        return

    pre_download_binary()

    logger.info("-" * 60)
    logger.info("ComfyUI-CoreAI ready.")
    logger.info("Nodes: CoreAIDepthEstimation, CoreAIObjectDetection,")
    logger.info("       CoreAIVisionLanguage")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Allow running directly: python install.py
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    on_install()
