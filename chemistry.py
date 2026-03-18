"""
chemistry.py — Chemistry Instruction Designer

Reorganises extracted TechnicalAtoms into a layman-friendly chemistry
instruction format based on five design principles:

1. Kit Mental Model      — Physical zones (Setup, Reagents, Reaction, Waste)
2. Sensory Checkpoints   — Visual / auditory / tactile anchors
3. Action-Result Syntax  — Strict action → expected observation per step
4. MIME-type Hardening   — Error handling, safe-pause points, path logic
5. Safety as Default     — Embedded PPE, Red Box warnings for hazardous steps
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from engine import TechnicalAtom, TaxonomyCategory, category_to_section


# ---------------------------------------------------------------------------
# 1. Kit Zones
# ---------------------------------------------------------------------------

class KitZone(str, Enum):
    """Physical zones that organise a chemistry workspace."""
    ZONE_A_SETUP = "Zone A: The Setup (Dry)"
    ZONE_B_REAGENTS = "Zone B: The Reagents (Cold / Stable)"
    ZONE_C_REACTION = "Zone C: The Reaction (Active)"
    ZONE_D_WASTE = "Zone D: The Waste (Neutralisation)"


# Map taxonomy categories → kit zones
_CATEGORY_ZONE_MAP: dict[str, KitZone] = {
    TaxonomyCategory.TOOLS_EQUIPMENT.value: KitZone.ZONE_A_SETUP,
    TaxonomyCategory.SAFETY_COMPLIANCE.value: KitZone.ZONE_A_SETUP,
    TaxonomyCategory.PERSONNEL_CERTIFICATIONS.value: KitZone.ZONE_A_SETUP,
    TaxonomyCategory.ENVIRONMENTAL_CONDITIONS.value: KitZone.ZONE_A_SETUP,
    TaxonomyCategory.MATERIALS.value: KitZone.ZONE_B_REAGENTS,
    TaxonomyCategory.CONSUMABLES.value: KitZone.ZONE_B_REAGENTS,
    TaxonomyCategory.DIMENSIONS_TOLERANCES.value: KitZone.ZONE_B_REAGENTS,
    TaxonomyCategory.PROCEDURAL_STEPS.value: KitZone.ZONE_C_REACTION,
    TaxonomyCategory.THERMAL.value: KitZone.ZONE_C_REACTION,
    TaxonomyCategory.FLUID_HYDRAULIC.value: KitZone.ZONE_C_REACTION,
    TaxonomyCategory.QUALITY_INSPECTION.value: KitZone.ZONE_C_REACTION,
    TaxonomyCategory.LOGISTICS_STORAGE.value: KitZone.ZONE_D_WASTE,
}

# Zone descriptions shown as preamble
ZONE_DESCRIPTIONS: dict[KitZone, str] = {
    KitZone.ZONE_A_SETUP: (
        "Gather all tools and personal protective equipment (PPE) before "
        "opening any chemical containers. Confirm everything is present."
    ),
    KitZone.ZONE_B_REAGENTS: (
        "All chemicals in their starting state. Verify labels, expiry dates, "
        "and quantities before proceeding."
    ),
    KitZone.ZONE_C_REACTION: (
        "The active workspace where mixing and reactions occur. Follow each "
        "step in order — do NOT skip ahead."
    ),
    KitZone.ZONE_D_WASTE: (
        "Specific neutralisation and disposal instructions. Complete these "
        "BEFORE leaving the workspace."
    ),
}


def zone_for_category(category_value: str) -> KitZone:
    """Map a taxonomy category to its kit zone."""
    return _CATEGORY_ZONE_MAP.get(category_value, KitZone.ZONE_C_REACTION)


def organise_by_zones(
    taxonomy: dict[str, list[TechnicalAtom]],
) -> dict[KitZone, list[TechnicalAtom]]:
    """Reorganise a full taxonomy into Kit Zones."""
    zones: dict[KitZone, list[TechnicalAtom]] = {z: [] for z in KitZone}
    for _section, atoms in taxonomy.items():
        for atom in atoms:
            zone = zone_for_category(atom.canonical_category())
            zones[zone].append(atom)
    return zones


# ---------------------------------------------------------------------------
# 2. Sensory Checkpoints
# ---------------------------------------------------------------------------

class SensoryType(str, Enum):
    VISUAL = "Visual"
    AUDITORY = "Auditory"
    TACTILE = "Tactile"
    OLFACTORY = "Olfactory"


@dataclass
class SensoryCheckpoint:
    """A human-perceivable confirmation that a step succeeded."""
    sensory_type: SensoryType
    description: str
    icon: str = ""

    def __post_init__(self) -> None:
        _icons = {
            SensoryType.VISUAL: "eye",
            SensoryType.AUDITORY: "ear",
            SensoryType.TACTILE: "hand",
            SensoryType.OLFACTORY: "nose",
        }
        if not self.icon:
            self.icon = _icons.get(self.sensory_type, "")


# Keyword → sensory checkpoint inference
_SENSORY_KEYWORDS: list[tuple[str, SensoryType, str]] = [
    # Visual cues
    (r"colour|color|turns?\s+\w+|cloudy|clear|transparent|opaque|precipitate|crystall?",
     SensoryType.VISUAL, "Watch for a colour / clarity change."),
    (r"bubbl|effervesce|foam", SensoryType.VISUAL,
     "Bubbles or foam should appear on the surface."),
    # Auditory cues
    (r"fizz|hiss|pop|crack|sizzl", SensoryType.AUDITORY,
     "You should hear a slight fizzing, similar to opening a soda."),
    (r"boil|rolling\s+boil", SensoryType.AUDITORY,
     "A rolling boil will produce a steady bubbling sound."),
    # Tactile cues
    (r"exotherm|warm|hot|heat\s+generat", SensoryType.TACTILE,
     "The outside of the container will feel warm to the touch (like a coffee mug)."),
    (r"endotherm|cold|cool|chill", SensoryType.TACTILE,
     "The container will feel noticeably cold to the touch."),
    # Olfactory cues
    (r"odou?r|smell|fume|pungent|acrid", SensoryType.OLFACTORY,
     "A noticeable odour may indicate the reaction has started — ensure ventilation."),
]


def infer_sensory_checkpoints(text: str) -> list[SensoryCheckpoint]:
    """Scan text for keywords and return matching sensory checkpoints."""
    results: list[SensoryCheckpoint] = []
    lower = text.lower()
    seen: set[SensoryType] = set()
    for pattern, stype, description in _SENSORY_KEYWORDS:
        if stype not in seen and re.search(pattern, lower):
            results.append(SensoryCheckpoint(sensory_type=stype, description=description))
            seen.add(stype)
    return results


# ---------------------------------------------------------------------------
# 3. Action-Result Steps
# ---------------------------------------------------------------------------

@dataclass
class ActionResultStep:
    """A single step in the strict Action → Expected Result format."""
    step_number: int
    action: str
    expected_result: str
    safety_note: str = ""
    is_hazardous: bool = False
    sensory_checkpoints: list[SensoryCheckpoint] = field(default_factory=list)
    is_safe_pause_point: bool = False
    troubleshooting: str = ""


def build_action_result_steps(
    procedural_atoms: list[TechnicalAtom],
) -> list[ActionResultStep]:
    """Convert procedural TechnicalAtoms into ActionResultStep objects.

    The atom's *parameter* becomes the action, and its *value* becomes the
    expected result. Safety notes are pulled from atom.notes when present.
    """
    steps: list[ActionResultStep] = []
    for i, atom in enumerate(procedural_atoms, start=1):
        action = atom.parameter
        expected = atom.value

        # Detect hazardous steps
        hazard_keywords = (
            "acid", "base", "corrosive", "toxic", "flammable", "oxidis",
            "oxidiz", "caustic", "irritant", "carcinogen", "explosive",
            "reactive", "danger", "hazard", "burn", "ignit",
        )
        is_hazardous = any(kw in (action + expected).lower() for kw in hazard_keywords)

        # Detect safe pause points
        pause_keywords = ("stable", "room temperature", "cool down", "let stand",
                          "wait", "overnight", "settle", "rest")
        is_pause = any(kw in (action + expected).lower() for kw in pause_keywords)

        # Infer sensory checkpoints from the result text
        checkpoints = infer_sensory_checkpoints(expected)

        # Build troubleshooting hint from notes
        troubleshooting = ""
        if atom.notes:
            troubleshooting = atom.notes

        safety_note = ""
        if is_hazardous and atom.notes:
            safety_note = atom.notes

        steps.append(ActionResultStep(
            step_number=i,
            action=action,
            expected_result=expected,
            safety_note=safety_note,
            is_hazardous=is_hazardous,
            sensory_checkpoints=checkpoints,
            is_safe_pause_point=is_pause,
            troubleshooting=troubleshooting,
        ))
    return steps


# ---------------------------------------------------------------------------
# 4. Troubleshooting Sidebars (MIME-type hardening)
# ---------------------------------------------------------------------------

@dataclass
class TroubleshootingEntry:
    """An error-handling sidebar for a phase or step."""
    phase: str
    symptom: str
    probable_cause: str
    corrective_action: str
    severity: str = "warning"  # "warning" | "critical"


def generate_troubleshooting(
    steps: list[ActionResultStep],
) -> list[TroubleshootingEntry]:
    """Auto-generate troubleshooting entries from action-result steps."""
    entries: list[TroubleshootingEntry] = []
    for step in steps:
        if step.troubleshooting:
            entries.append(TroubleshootingEntry(
                phase=f"Step {step.step_number}",
                symptom=f"Unexpected result at: {step.action[:60]}",
                probable_cause=step.troubleshooting,
                corrective_action="Do not proceed. Review the step and retry.",
                severity="critical" if step.is_hazardous else "warning",
            ))
        if step.is_hazardous:
            entries.append(TroubleshootingEntry(
                phase=f"Step {step.step_number}",
                symptom="Unexpected colour, odour, or excessive heat",
                probable_cause="Possible incorrect reagent quantity or contamination.",
                corrective_action=(
                    "STOP immediately. Do not add more reagents. "
                    "Allow the mixture to cool, then consult the safety data sheet."
                ),
                severity="critical",
            ))
    return entries


# ---------------------------------------------------------------------------
# 5. Safety Embedding
# ---------------------------------------------------------------------------

_PPE_KEYWORDS: dict[str, str] = {
    "gloves": "nitrile gloves",
    "goggles": "splash-proof safety goggles",
    "safety glasses": "safety glasses",
    "lab coat": "a buttoned lab coat",
    "face shield": "a full face shield",
    "fume hood": "a fume hood",
    "respirator": "an approved respirator",
    "apron": "a chemical-resistant apron",
}


def embed_safety_in_action(action: str, safety_atoms: list[TechnicalAtom]) -> str:
    """Rewrite an action string to embed relevant PPE reminders inline.

    Instead of: "Pick up the flask."
    Returns:    "While wearing your nitrile gloves, pick up the flask."
    """
    ppe_items: list[str] = []
    all_safety_text = " ".join(
        f"{a.parameter} {a.value}" for a in safety_atoms
    ).lower()
    for keyword, readable in _PPE_KEYWORDS.items():
        if keyword in all_safety_text:
            ppe_items.append(readable)

    if not ppe_items:
        return action

    # Only prepend if the action doesn't already mention PPE
    action_lower = action.lower()
    if any(kw in action_lower for kw in _PPE_KEYWORDS):
        return action

    ppe_str = " and ".join(ppe_items[:3])  # cap at 3 to avoid verbose
    # Lowercase the first letter of the original action for natural flow
    if action and action[0].isupper():
        action_rewritten = action[0].lower() + action[1:]
    else:
        action_rewritten = action
    return f"While wearing your {ppe_str}, {action_rewritten}"


# ---------------------------------------------------------------------------
# Full Chemistry Instruction Builder
# ---------------------------------------------------------------------------

@dataclass
class ChemistryInstructionSet:
    """Complete set of chemistry instructions for export / display."""
    title: str
    zones: dict[KitZone, list[TechnicalAtom]]
    steps: list[ActionResultStep]
    troubleshooting: list[TroubleshootingEntry]
    safe_pause_indices: list[int]


def build_chemistry_instructions(
    taxonomy: dict[str, list[TechnicalAtom]],
    title: str = "Chemistry Procedure",
) -> ChemistryInstructionSet:
    """Build a complete chemistry instruction set from a taxonomy.

    This is the main entry point: it orchestrates zones, action-result
    steps, safety embedding, sensory checkpoints, and troubleshooting.
    """
    # 1. Organise into kit zones
    zones = organise_by_zones(taxonomy)

    # 2. Collect safety atoms for embedding
    safety_atoms = zones.get(KitZone.ZONE_A_SETUP, [])
    safety_only = [a for a in safety_atoms
                   if a.canonical_category() == TaxonomyCategory.SAFETY_COMPLIANCE.value]

    # 3. Build action-result steps from procedural atoms
    procedural = zones.get(KitZone.ZONE_C_REACTION, [])
    proc_atoms = [a for a in procedural
                  if a.canonical_category() == TaxonomyCategory.PROCEDURAL_STEPS.value]
    steps = build_action_result_steps(proc_atoms)

    # 4. Embed safety into step actions
    for step in steps:
        step.action = embed_safety_in_action(step.action, safety_only)

    # 5. Generate troubleshooting
    troubleshooting = generate_troubleshooting(steps)

    # 6. Identify safe pause points
    safe_pause_indices = [s.step_number for s in steps if s.is_safe_pause_point]

    return ChemistryInstructionSet(
        title=title,
        zones=zones,
        steps=steps,
        troubleshooting=troubleshooting,
        safe_pause_indices=safe_pause_indices,
    )


# ---------------------------------------------------------------------------
# Detection: is this taxonomy chemistry-related?
# ---------------------------------------------------------------------------

_CHEMISTRY_SIGNALS = {
    "reagent", "reactant", "solvent", "solution", "acid", "base", "ph",
    "molar", "molarity", "titrat", "pipette", "beaker", "flask",
    "distill", "precipitat", "catalyst", "substrate", "buffer",
    "centrifug", "chromatograph", "spectro", "volumetric", "analyte",
    "dilut", "concentration", "stoichiometr", "equilibrium",
    "exotherm", "endotherm", "neutralis", "neutraliz", "oxidis",
    "oxidiz", "reduc", "synthesis", "compound", "chemical",
    "laboratory", "lab coat", "fume hood", "burette", "erlenmeyer",
    "boiling point", "melting point", "evaporat", "crystallis",
    "crystalliz", "filtrat", "decant",
}


def is_chemistry_taxonomy(taxonomy: dict[str, list[TechnicalAtom]]) -> bool:
    """Heuristic: returns True if the taxonomy looks chemistry-related.

    Scans atom parameters and values for chemistry-specific keywords.
    Triggers if >= 3 distinct chemistry signals are found.
    """
    hits: set[str] = set()
    for atoms in taxonomy.values():
        for atom in atoms:
            text = f"{atom.parameter} {atom.value}".lower()
            for signal in _CHEMISTRY_SIGNALS:
                if signal in text:
                    hits.add(signal)
                    if len(hits) >= 3:
                        return True
    return False
