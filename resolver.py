"""
resolver.py  –  The Operational Architect
Conflict Detection & Resolution Engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engine import TechnicalAtom

# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────


@dataclass
class Conflict:
    """A detected conflict between atoms from different source files."""

    conflict_id: str
    category: str
    parameter: str
    # Each entry: (source_file, value, full_atom)
    candidates: List[Tuple[str, str, TechnicalAtom]] = field(default_factory=list)
    resolved: bool = False
    resolved_atom: Optional[TechnicalAtom] = None

    def short_label(self) -> str:
        return f"[{self.category}] {self.parameter}"


@dataclass
class ResolutionState:
    """Tracks all detected conflicts and their resolution status."""

    conflicts: List[Conflict] = field(default_factory=list)

    @property
    def all_resolved(self) -> bool:
        return all(c.resolved for c in self.conflicts)

    @property
    def unresolved_count(self) -> int:
        return sum(1 for c in self.conflicts if not c.resolved)

    @property
    def total_count(self) -> int:
        return len(self.conflicts)


# ──────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────


def detect_conflicts(all_atoms: List[TechnicalAtom]) -> ResolutionState:
    """
    Scan all extracted atoms and group any that share (category, parameter)
    but carry different values from different source files.
    """
    # Key → list of atoms
    index: Dict[Tuple[str, str], List[TechnicalAtom]] = {}
    for atom in all_atoms:
        key = (atom.category.strip(), atom.parameter.strip().lower())
        index.setdefault(key, []).append(atom)

    state = ResolutionState()
    conflict_counter = 0

    for (category, parameter), atoms in index.items():
        # Only flag if there are multiple distinct values from different files
        unique_vals = {a.value.strip() for a in atoms}
        unique_files = {a.source_file for a in atoms}

        if len(unique_vals) > 1 and len(unique_files) > 1:
            conflict_counter += 1
            cid = f"CONFLICT_{conflict_counter:03d}"
            candidates = [(a.source_file, a.value, a) for a in atoms]
            state.conflicts.append(
                Conflict(
                    conflict_id=cid,
                    category=category,
                    parameter=atoms[0].parameter,  # preserve original casing
                    candidates=candidates,
                )
            )

    return state


# ──────────────────────────────────────────────
# Resolution
# ──────────────────────────────────────────────


def resolve_conflict(conflict: Conflict, chosen_index: int) -> TechnicalAtom:
    """
    Mark a conflict as resolved using the candidate at chosen_index.
    Returns the winning TechnicalAtom.
    """
    _, _, winning_atom = conflict.candidates[chosen_index]
    conflict.resolved = True
    conflict.resolved_atom = winning_atom
    return winning_atom


def apply_resolutions(
    taxonomy: Dict[str, List[TechnicalAtom]],
    state: ResolutionState,
) -> Dict[str, List[TechnicalAtom]]:
    """
    Return a new taxonomy where conflicting atoms have been replaced
    by the user-selected winners.  Atoms from losing files are dropped.
    """
    # Build a lookup: (category, parameter_lower) → winning atom
    winners: Dict[Tuple[str, str], TechnicalAtom] = {}
    losers: Dict[Tuple[str, str], set] = {}  # key → set of losing (file, value) pairs

    for conflict in state.conflicts:
        if not conflict.resolved or conflict.resolved_atom is None:
            continue
        key = (conflict.category, conflict.parameter.strip().lower())
        winners[key] = conflict.resolved_atom
        losing_vals = {
            (src, val)
            for src, val, _ in conflict.candidates
            if (src, val)
            != (conflict.resolved_atom.source_file, conflict.resolved_atom.value)
        }
        losers[key] = losing_vals

    resolved_taxonomy: Dict[str, List[TechnicalAtom]] = {}
    for section, atoms in taxonomy.items():
        filtered: List[TechnicalAtom] = []
        for atom in atoms:
            key = (atom.category, atom.parameter.strip().lower())
            if key in winners:
                # Replace with winner (add only once)
                winner = winners[key]
                if winner not in filtered:
                    filtered.append(winner)
            elif key in losers and (atom.source_file, atom.value) in losers[key]:
                # This is a losing candidate – drop it
                pass
            else:
                filtered.append(atom)
        resolved_taxonomy[section] = filtered

    return resolved_taxonomy


# ──────────────────────────────────────────────
# Flat atom list from resolved taxonomy
# ──────────────────────────────────────────────


def flatten_taxonomy(taxonomy: Dict[str, List[TechnicalAtom]]) -> List[TechnicalAtom]:
    """Flatten taxonomy dict into a single ordered list of atoms."""
    result: List[TechnicalAtom] = []
    for section in taxonomy:
        result.extend(taxonomy[section])
    return result
