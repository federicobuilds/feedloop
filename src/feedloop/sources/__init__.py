"""Slot implementations over concrete stores. ``filesystem`` serves a media folder."""
from feedloop.sources.filesystem import FilesystemSource, TextHashEncoder, build_text_hash_space

__all__ = ["FilesystemSource", "TextHashEncoder", "build_text_hash_space"]
