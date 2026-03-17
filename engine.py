"""
engine.py  –  Core data structures for the Operational Architect
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TechnicalAtom:
    """A single extracted technical parameter from a source document."""

    category: str
    parameter: str
    value: str
    source_file: str
