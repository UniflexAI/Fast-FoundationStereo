"""Fast-FoundationStereo plugins for TensorRT."""

from .gwc_plugin import register_gwc_plugin, GwcVolumePlugin

__all__ = ["register_gwc_plugin", "GwcVolumePlugin"]
