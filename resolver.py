"""
resolver.py — Conflict detection and resolution for TechnicalAtom sets.

Conflict detection keys on (category, parameter_lowercase) so two files calling
the same spec by the same name but with different values will always be caught.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from engine import TechnicalAtom, normalise_category


# ---------------------------------------------------------------------------
# Conflict dataclass
# ---------------------------------------------------------------------------

@dataclass
class Conflict:
    """Represents a parameter conflict between two or more atoms."""

    category: str
    parameter: str
    atoms: list[TechnicalAtom] = field(default_factory=list)
    resolution: str | None = None  # "keep_first", "keep_last", "manual:<value>", …
    resolved_atom: TechnicalAtom | None = None

    @property
    def key(self) -> str:
        return f"{self.category}||{self.parameter.strip().lower()}"

    @property
    def is_resolved(self) -> bool:
        return self.resolved_atom is not None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_conflicts(atoms: list[TechnicalAtom]) -> list[Conflict]:
    """Find all parameter-level conflicts across atoms.

    Two atoms conflict when they share the same canonical category and
    lower-cased parameter but differ in value.
    """
    buckets: dict[str, list[TechnicalAtom]] = defaultdict(list)
    for atom in atoms:
        buckets[atom.conflict_key()].append(atom)

    conflicts: list[Conflict] = []
    for key, group in buckets.items():
        if len(group) < 2:
            continue
        # Check that at least two distinct values exist
        values = {a.value.strip().lower() for a in group}
        if len(values) < 2:
            continue
        cat, param = key.split("||", 1)
        conflicts.append(Conflict(category=cat, parameter=param, atoms=group))

    return conflicts


# ---------------------------------------------------------------------------
# Resolution strategies
# ---------------------------------------------------------------------------

ResolutionStrategy = Literal[
    "keep_first",
    "keep_last",
    "keep_highest_confidence",
    "manual",
]


def resolve_conflict(
    conflict: Conflict,
    strategy: ResolutionStrategy = "keep_highest_confidence",
    manual_value: str | None = None,
) -> Conflict:
    """Apply a resolution strategy to a Conflict and return it updated."""
    if not conflict.atoms:
        return conflict

    if strategy == "keep_first":
        winner = conflict.atoms[0]
    elif strategy == "keep_last":
        winner = conflict.atoms[-1]
    elif strategy == "keep_highest_confidence":
        winner = max(conflict.atoms, key=lambda a: a.confidence)
    elif strategy == "manual":
        winner = conflict.atoms[0].model_copy()
        if manual_value is not None:
            winner.value = manual_value
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    conflict.resolved_atom = winner
    conflict.resolution = (
        f"manual:{manual_value}" if strategy == "manual" else strategy
    )
    return conflict


# ---------------------------------------------------------------------------
# Apply resolutions
# ---------------------------------------------------------------------------

def apply_resolutions(
    atoms: list[TechnicalAtom],
    conflicts: list[Conflict],
) -> list[TechnicalAtom]:
    """Return a new atom list with conflicts replaced by their resolved atoms.

    Non-conflicting atoms are kept as-is.  For each resolved conflict the
    winning atom replaces all members of that conflict group.
    """
    # Build a set of conflict keys that have been resolved
    resolved_keys: dict[str, TechnicalAtom] = {}
    conflicting_ids: set[int] = set()

    for c in conflicts:
        if not c.is_resolved:
            continue
        resolved_keys[c.key] = c.resolved_atom  # type: ignore[assignment]
        for a in c.atoms:
            conflicting_ids.add(id(a))

    result: list[TechnicalAtom] = []
    seen_keys: set[str] = set()

    for atom in atoms:
        if id(atom) in conflicting_ids:
            key = atom.conflict_key()
            if key in resolved_keys and key not in seen_keys:
                result.append(resolved_keys[key])
                seen_keys.add(key)
            # Skip other members of the conflict group
            continue
        result.append(atom)

    return result
