"""Shared filesystem paths used across the package."""
from __future__ import annotations
import pathlib

TEMPLATES_DIR = pathlib.Path(__file__).resolve().parent / "templates"
