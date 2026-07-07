"""
ComfyUI-CoreAI — custom node entry point.

ComfyUI loads custom_nodes/<name>/__init__.py and expects
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS at module scope.
The actual package lives in comfyui_coreai/.
"""
from comfyui_coreai import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, __version__

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
