"""
ComfyUI-CoreAI — custom node entry point.

ComfyUI loads custom_nodes/<name>/__init__.py via importlib.spec_from_file_location,
which does NOT add the custom node's directory to sys.path. We must do that
ourselves before importing the comfyui_coreai package.
"""
import os
import sys

# Add this directory to sys.path so `comfyui_coreai` is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from comfyui_coreai import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, __version__  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
