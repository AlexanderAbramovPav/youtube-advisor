"""Slug + name helpers."""


def short_slug(name: str) -> str:
    """Strip a trailing '-advisor' suffix from an advisor directory name."""
    suffix = "-advisor"
    return name[: -len(suffix)] if name.endswith(suffix) else name
