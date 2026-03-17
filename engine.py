"""
engine.py — TechnicalAtom Pydantic model · OperationalBrain (extraction + hoisting + taxonomy)
20-point taxonomy with alias map for normalising category names.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 20-point taxonomy
# ---------------------------------------------------------------------------

class TaxonomyCategory(str, Enum):
    MATERIALS = "Materials"
    TOOLS_EQUIPMENT = "Tools & Equipment"
    DIMENSIONS_TOLERANCES = "Dimensions & Tolerances"
    FASTENERS_HARDWARE = "Fasteners & Hardware"
    SURFACE_TREATMENT = "Surface Treatment / Finish"
    ELECTRICAL = "Electrical"
    THERMAL = "Thermal"
    FLUID_HYDRAULIC = "Fluid / Hydraulic"
    STRUCTURAL_LOADS = "Structural / Loads"
    SAFETY_COMPLIANCE = "Safety & Compliance"
    ENVIRONMENTAL_CONDITIONS = "Environmental Conditions"
    PROCEDURAL_STEPS = "Procedural Steps"
    QUALITY_INSPECTION = "Quality / Inspection"
    SOFTWARE_FIRMWARE = "Software / Firmware"
    COMMUNICATION_PROTOCOLS = "Communication Protocols"
    CONSUMABLES = "Consumables"
    PERSONNEL_CERTIFICATIONS = "Personnel & Certifications"
    LOGISTICS_STORAGE = "Logistics / Storage / Maintenance"
    DOCUMENTATION_REFERENCES = "Documentation & References"


# ---------------------------------------------------------------------------
# Ordered section names for display (19 operational + 1 meta)
# ---------------------------------------------------------------------------

TAXONOMY_SECTIONS = [
    "1. Materials",
    "2. Tools & Equipment",
    "3. Dimensions & Tolerances",
    "4. Fasteners & Hardware",
    "5. Surface Treatment / Finish",
    "6. Electrical",
    "7. Thermal",
    "8. Fluid / Hydraulic",
    "9. Structural / Loads",
    "10. Safety & Compliance",
    "11. Environmental Conditions",
    "12. Procedural Steps",
    "13. Quality / Inspection",
    "14. Software / Firmware",
    "15. Communication Protocols",
    "16. Consumables",
    "17. Personnel & Certifications",
    "18. Logistics / Storage / Maintenance",
    "19. Gap Audit",
    "20. Documentation & References",
]

# Map canonical TaxonomyCategory values → numbered section names
_CATEGORY_TO_SECTION: dict[str, str] = {
    TaxonomyCategory.MATERIALS.value: "1. Materials",
    TaxonomyCategory.TOOLS_EQUIPMENT.value: "2. Tools & Equipment",
    TaxonomyCategory.DIMENSIONS_TOLERANCES.value: "3. Dimensions & Tolerances",
    TaxonomyCategory.FASTENERS_HARDWARE.value: "4. Fasteners & Hardware",
    TaxonomyCategory.SURFACE_TREATMENT.value: "5. Surface Treatment / Finish",
    TaxonomyCategory.ELECTRICAL.value: "6. Electrical",
    TaxonomyCategory.THERMAL.value: "7. Thermal",
    TaxonomyCategory.FLUID_HYDRAULIC.value: "8. Fluid / Hydraulic",
    TaxonomyCategory.STRUCTURAL_LOADS.value: "9. Structural / Loads",
    TaxonomyCategory.SAFETY_COMPLIANCE.value: "10. Safety & Compliance",
    TaxonomyCategory.ENVIRONMENTAL_CONDITIONS.value: "11. Environmental Conditions",
    TaxonomyCategory.PROCEDURAL_STEPS.value: "12. Procedural Steps",
    TaxonomyCategory.QUALITY_INSPECTION.value: "13. Quality / Inspection",
    TaxonomyCategory.SOFTWARE_FIRMWARE.value: "14. Software / Firmware",
    TaxonomyCategory.COMMUNICATION_PROTOCOLS.value: "15. Communication Protocols",
    TaxonomyCategory.CONSUMABLES.value: "16. Consumables",
    TaxonomyCategory.PERSONNEL_CERTIFICATIONS.value: "17. Personnel & Certifications",
    TaxonomyCategory.LOGISTICS_STORAGE.value: "18. Logistics / Storage / Maintenance",
    TaxonomyCategory.DOCUMENTATION_REFERENCES.value: "20. Documentation & References",
}


# Alias map — maps common alternate names/abbreviations → canonical category
ALIAS_MAP: dict[str, TaxonomyCategory] = {
    # Materials
    "material": TaxonomyCategory.MATERIALS,
    "materials": TaxonomyCategory.MATERIALS,
    "raw material": TaxonomyCategory.MATERIALS,
    "raw materials": TaxonomyCategory.MATERIALS,
    "substrate": TaxonomyCategory.MATERIALS,
    "alloy": TaxonomyCategory.MATERIALS,
    "alloys": TaxonomyCategory.MATERIALS,
    "compound": TaxonomyCategory.MATERIALS,
    # Tools & Equipment
    "tools": TaxonomyCategory.TOOLS_EQUIPMENT,
    "tool": TaxonomyCategory.TOOLS_EQUIPMENT,
    "equipment": TaxonomyCategory.TOOLS_EQUIPMENT,
    "tooling": TaxonomyCategory.TOOLS_EQUIPMENT,
    "instruments": TaxonomyCategory.TOOLS_EQUIPMENT,
    "apparatus": TaxonomyCategory.TOOLS_EQUIPMENT,
    # Dimensions & Tolerances
    "dimensions": TaxonomyCategory.DIMENSIONS_TOLERANCES,
    "tolerances": TaxonomyCategory.DIMENSIONS_TOLERANCES,
    "tolerance": TaxonomyCategory.DIMENSIONS_TOLERANCES,
    "measurements": TaxonomyCategory.DIMENSIONS_TOLERANCES,
    "geometry": TaxonomyCategory.DIMENSIONS_TOLERANCES,
    "specs": TaxonomyCategory.DIMENSIONS_TOLERANCES,
    # Fasteners & Hardware
    "fasteners": TaxonomyCategory.FASTENERS_HARDWARE,
    "fastener": TaxonomyCategory.FASTENERS_HARDWARE,
    "hardware": TaxonomyCategory.FASTENERS_HARDWARE,
    "bolts": TaxonomyCategory.FASTENERS_HARDWARE,
    "screws": TaxonomyCategory.FASTENERS_HARDWARE,
    "rivets": TaxonomyCategory.FASTENERS_HARDWARE,
    # Surface Treatment / Finish
    "surface treatment": TaxonomyCategory.SURFACE_TREATMENT,
    "finish": TaxonomyCategory.SURFACE_TREATMENT,
    "coating": TaxonomyCategory.SURFACE_TREATMENT,
    "plating": TaxonomyCategory.SURFACE_TREATMENT,
    "paint": TaxonomyCategory.SURFACE_TREATMENT,
    "anodize": TaxonomyCategory.SURFACE_TREATMENT,
    "anodizing": TaxonomyCategory.SURFACE_TREATMENT,
    # Electrical
    "electrical": TaxonomyCategory.ELECTRICAL,
    "wiring": TaxonomyCategory.ELECTRICAL,
    "voltage": TaxonomyCategory.ELECTRICAL,
    "circuit": TaxonomyCategory.ELECTRICAL,
    "power": TaxonomyCategory.ELECTRICAL,
    # Thermal
    "thermal": TaxonomyCategory.THERMAL,
    "temperature": TaxonomyCategory.THERMAL,
    "heat": TaxonomyCategory.THERMAL,
    "cooling": TaxonomyCategory.THERMAL,
    # Fluid / Hydraulic
    "fluid": TaxonomyCategory.FLUID_HYDRAULIC,
    "hydraulic": TaxonomyCategory.FLUID_HYDRAULIC,
    "pneumatic": TaxonomyCategory.FLUID_HYDRAULIC,
    "pressure": TaxonomyCategory.FLUID_HYDRAULIC,
    "flow": TaxonomyCategory.FLUID_HYDRAULIC,
    # Structural / Loads
    "structural": TaxonomyCategory.STRUCTURAL_LOADS,
    "loads": TaxonomyCategory.STRUCTURAL_LOADS,
    "load": TaxonomyCategory.STRUCTURAL_LOADS,
    "stress": TaxonomyCategory.STRUCTURAL_LOADS,
    "strain": TaxonomyCategory.STRUCTURAL_LOADS,
    "fatigue": TaxonomyCategory.STRUCTURAL_LOADS,
    # Safety & Compliance
    "safety": TaxonomyCategory.SAFETY_COMPLIANCE,
    "compliance": TaxonomyCategory.SAFETY_COMPLIANCE,
    "regulatory": TaxonomyCategory.SAFETY_COMPLIANCE,
    "osha": TaxonomyCategory.SAFETY_COMPLIANCE,
    "hazard": TaxonomyCategory.SAFETY_COMPLIANCE,
    # Environmental Conditions
    "environmental": TaxonomyCategory.ENVIRONMENTAL_CONDITIONS,
    "environment": TaxonomyCategory.ENVIRONMENTAL_CONDITIONS,
    "climate": TaxonomyCategory.ENVIRONMENTAL_CONDITIONS,
    "weather": TaxonomyCategory.ENVIRONMENTAL_CONDITIONS,
    "humidity": TaxonomyCategory.ENVIRONMENTAL_CONDITIONS,
    # Procedural Steps
    "procedural": TaxonomyCategory.PROCEDURAL_STEPS,
    "procedure": TaxonomyCategory.PROCEDURAL_STEPS,
    "steps": TaxonomyCategory.PROCEDURAL_STEPS,
    "process": TaxonomyCategory.PROCEDURAL_STEPS,
    "workflow": TaxonomyCategory.PROCEDURAL_STEPS,
    "instructions": TaxonomyCategory.PROCEDURAL_STEPS,
    # Quality / Inspection
    "quality": TaxonomyCategory.QUALITY_INSPECTION,
    "inspection": TaxonomyCategory.QUALITY_INSPECTION,
    "qc": TaxonomyCategory.QUALITY_INSPECTION,
    "qa": TaxonomyCategory.QUALITY_INSPECTION,
    "testing": TaxonomyCategory.QUALITY_INSPECTION,
    "ndt": TaxonomyCategory.QUALITY_INSPECTION,
    # Software / Firmware
    "software": TaxonomyCategory.SOFTWARE_FIRMWARE,
    "firmware": TaxonomyCategory.SOFTWARE_FIRMWARE,
    "code": TaxonomyCategory.SOFTWARE_FIRMWARE,
    "plc": TaxonomyCategory.SOFTWARE_FIRMWARE,
    # Communication Protocols
    "communication": TaxonomyCategory.COMMUNICATION_PROTOCOLS,
    "protocol": TaxonomyCategory.COMMUNICATION_PROTOCOLS,
    "protocols": TaxonomyCategory.COMMUNICATION_PROTOCOLS,
    "can bus": TaxonomyCategory.COMMUNICATION_PROTOCOLS,
    "modbus": TaxonomyCategory.COMMUNICATION_PROTOCOLS,
    # Consumables
    "consumables": TaxonomyCategory.CONSUMABLES,
    "consumable": TaxonomyCategory.CONSUMABLES,
    "lubricant": TaxonomyCategory.CONSUMABLES,
    "adhesive": TaxonomyCategory.CONSUMABLES,
    "sealant": TaxonomyCategory.CONSUMABLES,
    # Personnel & Certifications
    "personnel": TaxonomyCategory.PERSONNEL_CERTIFICATIONS,
    "certifications": TaxonomyCategory.PERSONNEL_CERTIFICATIONS,
    "certification": TaxonomyCategory.PERSONNEL_CERTIFICATIONS,
    "training": TaxonomyCategory.PERSONNEL_CERTIFICATIONS,
    "qualifications": TaxonomyCategory.PERSONNEL_CERTIFICATIONS,
    # Logistics / Storage / Maintenance
    "logistics": TaxonomyCategory.LOGISTICS_STORAGE,
    "storage": TaxonomyCategory.LOGISTICS_STORAGE,
    "warehouse": TaxonomyCategory.LOGISTICS_STORAGE,
    "shipping": TaxonomyCategory.LOGISTICS_STORAGE,
    "handling": TaxonomyCategory.LOGISTICS_STORAGE,
    "maintenance": TaxonomyCategory.LOGISTICS_STORAGE,
    "schedule": TaxonomyCategory.LOGISTICS_STORAGE,
    "preventive maintenance": TaxonomyCategory.LOGISTICS_STORAGE,
    "pm": TaxonomyCategory.LOGISTICS_STORAGE,
    "overhaul": TaxonomyCategory.LOGISTICS_STORAGE,
    # Documentation & References
    "documentation": TaxonomyCategory.DOCUMENTATION_REFERENCES,
    "references": TaxonomyCategory.DOCUMENTATION_REFERENCES,
    "docs": TaxonomyCategory.DOCUMENTATION_REFERENCES,
    "drawings": TaxonomyCategory.DOCUMENTATION_REFERENCES,
    "manuals": TaxonomyCategory.DOCUMENTATION_REFERENCES,
    "sop": TaxonomyCategory.DOCUMENTATION_REFERENCES,
}


def normalise_category(raw: str) -> TaxonomyCategory:
    """Resolve a raw category string to its canonical TaxonomyCategory."""
    key = raw.strip().lower()
    # Direct enum value match
    for cat in TaxonomyCategory:
        if cat.value.lower() == key:
            return cat
    # Alias lookup
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]
    # Fuzzy: check if any alias is a substring
    for alias, cat in ALIAS_MAP.items():
        if alias in key or key in alias:
            return cat
    # Fallback — return Documentation & References as catch-all
    return TaxonomyCategory.DOCUMENTATION_REFERENCES


def category_to_section(category_value: str) -> str:
    """Map a canonical category value to its numbered section name."""
    return _CATEGORY_TO_SECTION.get(category_value, "20. Documentation & References")


# ---------------------------------------------------------------------------
# TechnicalAtom — the atomic unit of extracted technical knowledge
# ---------------------------------------------------------------------------

class TechnicalAtom(BaseModel):
    """A single atomic piece of technical specification extracted from a document."""

    category: str = Field(
        ..., description="Taxonomy category (one of the 20-point taxonomy values)"
    )
    parameter: str = Field(
        ..., description="The specific parameter or attribute name"
    )
    value: str = Field(
        ..., description="The value or specification for this parameter"
    )
    unit: Optional[str] = Field(
        None, description="Unit of measurement, if applicable"
    )
    source_file: str = Field(
        ..., description="Name of the source document"
    )
    page_or_section: Optional[str] = Field(
        None, description="Page number or section reference"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Extraction confidence score (0.0 – 1.0)"
    )
    notes: Optional[str] = Field(
        None, description="Additional context or caveats"
    )
    is_prerequisite: bool = Field(
        default=False,
        description="Whether this atom represents a prerequisite condition",
    )
    hoisted: bool = Field(
        default=False,
        description="Whether this atom was hoisted from a procedural step to its proper category",
    )

    def canonical_category(self) -> TaxonomyCategory:
        return normalise_category(self.category)

    def conflict_key(self) -> str:
        """Key used for conflict detection: (canonical_category, parameter_lower)."""
        return f"{self.canonical_category().value}||{self.parameter.strip().lower()}"


# ---------------------------------------------------------------------------
# LLM client helpers — provider-agnostic
# ---------------------------------------------------------------------------

def _call_anthropic(
    prompt: str,
    system: str,
    model: str,
    api_key: str,
    max_tokens: int = 4096,
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _call_openai(
    prompt: str,
    system: str,
    model: str,
    api_key: str,
    max_tokens: int = 4096,
) -> str:
    import openai

    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def call_llm(
    prompt: str,
    system: str,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str = "",
    max_tokens: int = 4096,
) -> str:
    """Dispatch a prompt to the chosen LLM provider and return raw text."""
    if provider.lower() == "anthropic":
        model = model or "claude-sonnet-4-20250514"
        return _call_anthropic(prompt, system, model, api_key, max_tokens)
    elif provider.lower() == "openai":
        model = model or "gpt-4o"
        return _call_openai(prompt, system, model, api_key, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# OperationalBrain — extraction + hoisting + taxonomy classification
# ---------------------------------------------------------------------------

TAXONOMY_LIST = "\n".join(f"- {c.value}" for c in TaxonomyCategory)

SYSTEM_PROMPT = f"""\
You are a meticulous technical-specification extraction engine.

## Taxonomy (19 categories)
{TAXONOMY_LIST}

## Rules
1. Extract every atomic technical fact from the document as a JSON object with
   these fields: category, parameter, value, unit (nullable), page_or_section
   (nullable), confidence (0-1), notes (nullable), is_prerequisite (bool),
   hoisted (bool).
2. **Hoisting rule**: If a procedural step references a specific tool, material,
   consumable, or piece of equipment, you MUST emit TWO atoms:
   - One atom for the procedural step itself (category = "Procedural Steps").
   - One atom for the referenced item under its proper category (e.g.
     "Tools & Equipment", "Materials", "Consumables") with hoisted = true.
3. **Prerequisite rule**: If a step or spec is a prerequisite for other work,
   set is_prerequisite = true.
4. Use the exact category names from the taxonomy above.
5. Return ONLY a JSON array of objects — no markdown fences, no commentary.
"""

EXTRACTION_PROMPT_TEMPLATE = """\
Source file: {filename}

--- DOCUMENT TEXT ---
{text}
--- END ---

Extract all technical atoms as a JSON array. Remember the hoisting and prerequisite rules.
"""


class OperationalBrain:
    """Orchestrates LLM-based extraction, hoisting enforcement, and taxonomy normalisation."""

    def __init__(
        self,
        provider: str = "Anthropic",
        model: str | None = None,
        api_key: str = "",
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def extract_atoms(self, text: str, filename: str) -> list[TechnicalAtom]:
        """Extract TechnicalAtoms from a document's text."""
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(filename=filename, text=text)
        raw = call_llm(
            prompt,
            SYSTEM_PROMPT,
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
        )
        atoms = self._parse_atoms(raw, filename)
        atoms = self._normalise(atoms)
        return atoms

    def organize_by_taxonomy(
        self, atoms: list[TechnicalAtom]
    ) -> Dict[str, List[TechnicalAtom]]:
        """Group atoms into numbered taxonomy sections for display."""
        grouped: Dict[str, List[TechnicalAtom]] = {
            section: [] for section in TAXONOMY_SECTIONS
        }
        for atom in atoms:
            cat = normalise_category(atom.category).value
            section = category_to_section(cat)
            if section in grouped:
                grouped[section].append(atom)
        return grouped

    # Aliases for backward compatibility
    extract = extract_atoms

    def classify_by_taxonomy(
        self, atoms: list[TechnicalAtom]
    ) -> dict[str, list[TechnicalAtom]]:
        """Group atoms by their canonical taxonomy category."""
        grouped: dict[str, list[TechnicalAtom]] = {
            cat.value: [] for cat in TaxonomyCategory
        }
        for atom in atoms:
            cat = normalise_category(atom.category).value
            grouped[cat].append(atom)
        return grouped

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_atoms(raw: str, source_file: str) -> list[TechnicalAtom]:
        """Parse the LLM JSON response into TechnicalAtom instances."""
        # Strip markdown fences if the LLM added them
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        cleaned = cleaned.strip("`").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find a JSON array in the response
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                return []

        atoms: list[TechnicalAtom] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            item.setdefault("source_file", source_file)
            try:
                atoms.append(TechnicalAtom(**item))
            except Exception:
                continue
        return atoms

    @staticmethod
    def _normalise(atoms: list[TechnicalAtom]) -> list[TechnicalAtom]:
        """Normalise category names to canonical taxonomy values."""
        for atom in atoms:
            atom.category = normalise_category(atom.category).value
        return atoms
