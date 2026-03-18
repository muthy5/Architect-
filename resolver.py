"""
resolver.py — Conflict detection and resolution for TechnicalAtom sets.

Conflict detection keys on (category, parameter_lowercase) so two files calling
the same spec by the same name but with different values will always be caught.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

from engine import TechnicalAtom, normalise_category

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conflict dataclass
# ---------------------------------------------------------------------------

@dataclass
class Conflict:
    """Represents a parameter conflict between two or more atoms."""

    conflict_id: str
    category: str
    parameter: str
    atoms: list[TechnicalAtom] = field(default_factory=list)
    resolved: bool = False
    resolved_atom: TechnicalAtom | None = None

    @property
    def key(self) -> str:
        return f"{self.category}||{self.parameter.strip().lower()}"

    @property
    def candidates(self) -> list[tuple[str, str, TechnicalAtom]]:
        """Return list of (source_file, value, atom) tuples for UI display."""
        return [(a.source_file, a.value, a) for a in self.atoms]

    def short_label(self) -> str:
        """Short human-readable label for the conflict."""
        sources = ", ".join(dict.fromkeys(a.source_file for a in self.atoms))
        return f"{self.parameter} ({sources})"


# ---------------------------------------------------------------------------
# ResolutionState — tracks all conflicts for a session
# ---------------------------------------------------------------------------

@dataclass
class ResolutionState:
    """Container for all detected conflicts in a session."""

    conflicts: list[Conflict] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.conflicts)

    @property
    def unresolved_count(self) -> int:
        return sum(1 for c in self.conflicts if not c.resolved)

    @property
    def all_resolved(self) -> bool:
        return all(c.resolved for c in self.conflicts)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_conflicts(atoms: list[TechnicalAtom]) -> ResolutionState:
    """Find all parameter-level conflicts across atoms.

    Two atoms conflict when they share the same canonical category and
    lower-cased parameter but differ in value.
    Returns a ResolutionState containing all detected Conflict objects.
    """
    buckets: dict[str, list[TechnicalAtom]] = defaultdict(list)
    for atom in atoms:
        buckets[atom.conflict_key()].append(atom)

    conflicts: list[Conflict] = []
    conflict_num = 0
    for key, group in buckets.items():
        if len(group) < 2:
            continue
        # Check that at least two distinct values exist
        values = {a.value.strip().lower() for a in group}
        if len(values) < 2:
            continue
        cat, param = key.split("||", 1)
        conflict_num += 1
        conflicts.append(
            Conflict(
                conflict_id=f"C-{conflict_num:03d}",
                category=cat,
                parameter=param,
                atoms=group,
            )
        )

    return ResolutionState(conflicts=conflicts)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_conflict(conflict: Conflict, chosen_idx: int) -> None:
    """Resolve a conflict by selecting the candidate at chosen_idx."""
    if not conflict.atoms or chosen_idx < 0 or chosen_idx >= len(conflict.atoms):
        return
    conflict.resolved_atom = conflict.atoms[chosen_idx]
    conflict.resolved = True


# ---------------------------------------------------------------------------
# Apply resolutions to taxonomy
# ---------------------------------------------------------------------------

def apply_resolutions(
    taxonomy: dict[str, list[TechnicalAtom]],
    resolution_state: ResolutionState,
) -> dict[str, list[TechnicalAtom]]:
    """Return a new taxonomy dict with conflicts replaced by their resolved atoms.

    Non-conflicting atoms are kept as-is. For each resolved conflict the
    winning atom replaces all members of that conflict group.
    """
    # Build lookup of conflict keys → resolved atom
    conflict_keys: set[str] = set()
    resolved_by_key: dict[str, TechnicalAtom] = {}

    for c in resolution_state.conflicts:
        for a in c.atoms:
            conflict_keys.add(a.conflict_key())
        if not c.resolved or not c.resolved_atom:
            continue
        resolved_by_key[c.key] = c.resolved_atom

    # Warn about unresolved conflicts — these atoms will be dropped entirely
    unresolved = [
        c for c in resolution_state.conflicts
        if not c.resolved or not c.resolved_atom
    ]
    if unresolved:
        dropped_params = [c.short_label() for c in unresolved]
        msg = (
            f"{len(unresolved)} unresolved conflict(s) will cause atoms to be "
            f"dropped from the exported taxonomy: {', '.join(dropped_params)}"
        )
        log.warning(msg)
        warnings.warn(msg, stacklevel=2)

    result: dict[str, list[TechnicalAtom]] = {}
    seen_keys: set[str] = set()

    for section, atoms in taxonomy.items():
        new_atoms: list[TechnicalAtom] = []
        for atom in atoms:
            ckey = atom.conflict_key()
            if ckey in conflict_keys:
                if ckey in resolved_by_key and ckey not in seen_keys:
                    new_atoms.append(resolved_by_key[ckey])
                    seen_keys.add(ckey)
                # Skip other members of the conflict group
                continue
            new_atoms.append(atom)
        result[section] = new_atoms

    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def flatten_taxonomy(taxonomy: dict[str, list[TechnicalAtom]]) -> list[TechnicalAtom]:
    """Flatten a taxonomy dict into a single list of atoms."""
    atoms: list[TechnicalAtom] = []
    for section_atoms in taxonomy.values():
        atoms.extend(section_atoms)
    return atoms
