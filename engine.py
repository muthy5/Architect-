"""
engine.py — TechnicalAtom Pydantic model · OperationalBrain (extraction + hoisting + taxonomy)
20-point taxonomy with alias map for normalising category names.
Self-healing: LLM calls and JSON parsing auto-recover from failures using
learned strategies from previous runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

from self_healer import get_healer

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
    # Fuzzy: check if a multi-word alias appears in the key
    # Only match aliases with 4+ chars to avoid false positives from short aliases
    for alias, cat in ALIAS_MAP.items():
        if len(alias) >= 4 and alias in key:
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
    unit: str | None = Field(
        None, description="Unit of measurement, if applicable"
    )
    source_file: str = Field(
        ..., description="Name of the source document"
    )
    page_or_section: str | None = Field(
        None, description="Page number or section reference"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Extraction confidence score (0.0 – 1.0)"
    )
    notes: str | None = Field(
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
# Extraction diagnostics — surfaced to the UI so failures aren't silent
# ---------------------------------------------------------------------------


@dataclass
class FileExtractionDiag:
    """Diagnostic info for a single file's extraction attempt."""

    filename: str
    file_read_ok: bool = True
    file_read_error: str = ""
    text_length: int = 0
    llm_call_ok: bool = True
    llm_call_error: str = ""
    llm_raw_response_length: int = 0
    json_parse_ok: bool = True
    json_parse_error: str = ""
    json_cleanup_level: int = -1  # -1 = not attempted, 0-3 = cleanup level used
    items_in_json: int = 0
    atoms_constructed: int = 0
    atoms_skipped: int = 0
    skipped_reasons: list[str] = field(default_factory=list)
    atoms_after_normalise: int = 0
    warnings: list[str] = field(default_factory=list)

    def has_problems(self) -> bool:
        return (
            not self.file_read_ok
            or not self.llm_call_ok
            or not self.json_parse_ok
            or self.atoms_skipped > 0
            or (self.items_in_json > 0 and self.atoms_constructed == 0)
            or (self.text_length > 0 and self.atoms_after_normalise == 0)
        )

    def summary_lines(self) -> list[str]:
        """Return human-readable diagnostic lines."""
        lines: list[str] = []
        if not self.file_read_ok:
            lines.append(f"File read failed: {self.file_read_error}")
            return lines
        lines.append(f"Text extracted: {self.text_length:,} chars")
        if not self.llm_call_ok:
            lines.append(f"LLM call failed: {self.llm_call_error}")
            return lines
        lines.append(f"LLM response: {self.llm_raw_response_length:,} chars")
        if not self.json_parse_ok:
            lines.append(f"JSON parsing failed: {self.json_parse_error}")
            return lines
        if self.json_cleanup_level > 0:
            lines.append(
                f"JSON required cleanup level {self.json_cleanup_level} to parse"
            )
        lines.append(f"JSON items found: {self.items_in_json}")
        lines.append(f"Atoms constructed: {self.atoms_constructed}")
        if self.atoms_skipped > 0:
            lines.append(
                f"Atoms skipped (malformed): {self.atoms_skipped}"
            )
            for reason in self.skipped_reasons[:5]:
                lines.append(f"  - {reason}")
        lines.append(f"Atoms after normalisation: {self.atoms_after_normalise}")
        for w in self.warnings:
            lines.append(f"Warning: {w}")
        return lines


@dataclass
class ExtractionDiagnostics:
    """Collects diagnostics across all files in an extraction run."""

    file_diags: list[FileExtractionDiag] = field(default_factory=list)

    def add(self, diag: FileExtractionDiag) -> None:
        self.file_diags.append(diag)

    def total_atoms(self) -> int:
        return sum(d.atoms_after_normalise for d in self.file_diags)

    def total_skipped(self) -> int:
        return sum(d.atoms_skipped for d in self.file_diags)

    def files_with_problems(self) -> list[FileExtractionDiag]:
        return [d for d in self.file_diags if d.has_problems()]

    def has_any_problems(self) -> bool:
        return any(d.has_problems() for d in self.file_diags)


# ---------------------------------------------------------------------------
# LLM client helpers — provider-agnostic
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cost tiers — model recommendations per provider
# ---------------------------------------------------------------------------

COST_TIERS = {
    "anthropic": {
        "quality": {"model": "claude-sonnet-4-20250514", "label": "Claude Sonnet 4 (best quality)"},
        "balanced": {"model": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5 (good quality, 10x cheaper)"},
        "economy": {"model": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5 (lowest cost)"},
    },
    "openai": {
        "quality": {"model": "gpt-4o", "label": "GPT-4o (best quality)"},
        "balanced": {"model": "gpt-4o-mini", "label": "GPT-4o Mini (good quality, 15x cheaper)"},
        "economy": {"model": "gpt-4o-mini", "label": "GPT-4o Mini (lowest cost)"},
    },
}


def get_tier_model(provider: str, tier: str) -> str:
    """Return the default model for a given provider and cost tier."""
    provider_key = provider.strip().lower()
    tier_key = tier.strip().lower()
    return COST_TIERS.get(provider_key, {}).get(tier_key, {}).get(
        "model", "claude-sonnet-4-20250514"
    )


def estimate_max_tokens(input_text_len: int) -> int:
    """Scale max output tokens based on input size.

    Smaller documents need fewer output tokens. This avoids reserving
    16 K output tokens for a 500-word document, which lets the API
    pick a cheaper routing / reduces latency.
    """
    # Rough heuristic: 1 char ≈ 0.25 tokens; expect output ≈ 1.5x the atom density
    estimated_input_tokens = input_text_len // 4
    # Minimum 2048, cap at 16384
    return max(2048, min(16384, estimated_input_tokens * 2))


# ---------------------------------------------------------------------------
# File-hash response cache (session-scoped, avoids duplicate API calls)
# ---------------------------------------------------------------------------

_extraction_cache: dict[str, list[dict]] = {}


def _cache_key(text: str, filename: str, provider: str, model: str) -> str:
    """Create a deterministic cache key from document content + model config."""
    h = hashlib.sha256(f"{text}|{filename}|{provider}|{model}".encode()).hexdigest()
    return h


def get_cached_extraction(text: str, filename: str, provider: str, model: str):
    """Return cached atom dicts if available, else None."""
    key = _cache_key(text, filename, provider, model)
    return _extraction_cache.get(key)


def set_cached_extraction(text: str, filename: str, provider: str, model: str, atoms_raw: str):
    """Store raw LLM response in cache."""
    key = _cache_key(text, filename, provider, model)
    _extraction_cache[key] = atoms_raw


def clear_extraction_cache() -> int:
    """Clear the in-memory extraction cache. Returns the number of entries removed."""
    count = len(_extraction_cache)
    _extraction_cache.clear()
    return count


# ---------------------------------------------------------------------------
# Token usage tracking
# ---------------------------------------------------------------------------

class UsageStats:
    """Tracks cumulative token usage and estimated costs for a session."""

    # Approximate pricing per million tokens (USD) as of 2025
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
        "gpt-4o": {"input": 2.50, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.calls = 0
        self.cache_hits = 0

    def record(self, model: str, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.calls += 1
        pricing = self.PRICING.get(model, {"input": 3.0, "output": 15.0})
        self.total_cost_usd += (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )

    def record_cache_hit(self):
        self.cache_hits += 1

    def reset(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.calls = 0
        self.cache_hits = 0

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 6),
        }


# Global session stats instance
usage_stats = UsageStats()


def _call_anthropic(
    prompt: str,
    system: str,
    model: str,
    api_key: str,
    max_tokens: int = 16384,
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    # Use prompt caching: mark the system prompt as cacheable so it's
    # reused across multiple file extractions in the same session.
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )
    # Track token usage
    if hasattr(resp, "usage") and resp.usage:
        input_tok = getattr(resp.usage, "input_tokens", 0)
        output_tok = getattr(resp.usage, "output_tokens", 0)
        usage_stats.record(model, input_tok, output_tok)
    return resp.content[0].text


def _call_openai(
    prompt: str,
    system: str,
    model: str,
    api_key: str,
    max_tokens: int = 16384,
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
    # Track token usage
    if hasattr(resp, "usage") and resp.usage:
        input_tok = getattr(resp.usage, "prompt_tokens", 0)
        output_tok = getattr(resp.usage, "completion_tokens", 0)
        usage_stats.record(model, input_tok, output_tok)
    return resp.choices[0].message.content


def call_llm(
    prompt: str,
    system: str,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str = "",
    max_tokens: int = 16384,
    tier: str = "quality",
) -> str:
    """Dispatch a prompt to the chosen LLM provider and return raw text.

    Retries automatically on transient errors (rate-limit, timeout, overloaded).
    Self-healing: also records outcomes and applies learned recovery strategies
    (input truncation on token-limit errors, stronger JSON instructions, etc.).
    """
    healer = get_healer()

    if provider.lower() == "anthropic":
        resolved_model = model or get_tier_model("anthropic", tier)
        fn = _call_anthropic
    elif provider.lower() == "openai":
        resolved_model = model or get_tier_model("openai", tier)
        fn = _call_openai
    else:
        raise ValueError(f"Unknown provider: {provider}")

    context = {"provider": provider, "model": resolved_model}

    # -- Recovery handlers that modify args/kwargs before retry --
    def _reduce_input(args, kwargs, error):
        """Truncate prompt to ~60% to fit token limits."""
        p, s, m, k, mt = args
        truncated = p[: int(len(p) * 0.6)]
        truncated += "\n\n[...truncated for length — extract what you can from above...]"
        return (truncated, s, m, k, mt), kwargs

    def _add_json_instruction(args, kwargs, error):
        """Reinforce JSON output instruction in system prompt."""
        p, s, m, k, mt = args
        s += (
            "\n\nCRITICAL: Your previous response was not valid JSON. "
            "You MUST return ONLY a JSON array. No markdown, no commentary."
        )
        return (p, s, m, k, mt), kwargs

    recovery_handlers = {
        "reduce_input_size": _reduce_input,
        "add_json_instruction": _add_json_instruction,
    }

    return healer.execute_with_healing(
        operation="llm_call",
        func=fn,
        *(prompt, system, resolved_model, api_key, max_tokens),
        context=context,
        max_retries=3,
        recovery_handlers=recovery_handlers,
    )


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


def _coerce_atom_fields(item: dict) -> dict:
    """Coerce LLM-returned field values to the types TechnicalAtom expects.

    Handles common type mismatches: confidence as string, is_prerequisite as
    string, numeric values in string fields, and unexpected extra keys.
    """
    coerced = dict(item)
    # str fields — force to str
    for key in ("category", "parameter", "value", "unit",
                "source_file", "page_or_section", "notes"):
        if key in coerced and coerced[key] is not None:
            coerced[key] = str(coerced[key])
    # confidence → float
    if "confidence" in coerced:
        try:
            coerced["confidence"] = float(coerced["confidence"])
        except (ValueError, TypeError):
            coerced["confidence"] = 1.0
    # is_prerequisite → bool
    if "is_prerequisite" in coerced:
        v = coerced["is_prerequisite"]
        if isinstance(v, str):
            coerced["is_prerequisite"] = v.lower() in ("true", "1", "yes")
        else:
            coerced["is_prerequisite"] = bool(v)
    # Strip unexpected keys that cause TypeError on construction
    valid_keys = {
        "category", "parameter", "value", "unit", "source_file",
        "page_or_section", "confidence", "notes", "is_prerequisite",
    }
    coerced = {k: v for k, v in coerced.items() if k in valid_keys}
    return coerced


class OperationalBrain:
    """Orchestrates LLM-based extraction, hoisting enforcement, and taxonomy normalisation."""

    def __init__(
        self,
        provider: str = "Anthropic",
        model: str | None = None,
        api_key: str = "",
        tier: str = "quality",
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.tier = tier

    def extract_atoms(
        self,
        text: str,
        filename: str,
        diag: FileExtractionDiag | None = None,
    ) -> list[TechnicalAtom]:
        """Extract TechnicalAtoms from a document's text.

        Uses file-content caching to skip API calls for previously-extracted
        documents, and dynamic max_tokens to reduce cost on smaller files.
        When *diag* is provided, populates it with step-by-step diagnostic info.
        """
        effective_model = self.model or get_tier_model(self.provider, self.tier)

        if diag is not None:
            diag.text_length = len(text)

        # Check cache — skip API call if we've already extracted this exact content
        cached = get_cached_extraction(text, filename, self.provider, effective_model)
        if cached is not None:
            log.info("Cache hit for %s — skipping API call", filename)
            usage_stats.record_cache_hit()
            if diag is not None:
                diag.warnings.append("Using cached LLM response (no new API call)")
            atoms = self._parse_atoms(cached, filename, diag)
            atoms = self._normalise(atoms)
            if diag is not None:
                diag.atoms_after_normalise = len(atoms)
            return atoms

        prompt = EXTRACTION_PROMPT_TEMPLATE.format(filename=filename, text=text)
        max_tok = estimate_max_tokens(len(text))
        try:
            raw = call_llm(
                prompt,
                SYSTEM_PROMPT,
                provider=self.provider,
                model=self.model,
                api_key=self.api_key,
                max_tokens=max_tok,
                tier=self.tier,
            )
        except Exception as e:
            if diag is not None:
                diag.llm_call_ok = False
                diag.llm_call_error = f"{type(e).__name__}: {e}"
            raise

        # Cache the raw response for future re-uploads
        set_cached_extraction(text, filename, self.provider, effective_model, raw)

        atoms = self._parse_atoms(raw, filename, diag)
        atoms = self._normalise(atoms)
        if diag is not None:
            diag.atoms_after_normalise = len(atoms)
        return atoms

    def organize_by_taxonomy(
        self, atoms: list[TechnicalAtom]
    ) -> dict[str, list[TechnicalAtom]]:
        """Group atoms into numbered taxonomy sections for display."""
        grouped: dict[str, list[TechnicalAtom]] = {
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
    def _parse_atoms(
        raw: str,
        source_file: str,
        diag: FileExtractionDiag | None = None,
    ) -> list[TechnicalAtom]:
        """Parse the LLM JSON response into TechnicalAtom instances.

        Self-healing: applies progressively aggressive cleanup strategies
        when JSON parsing fails, and records outcomes for future learning.
        Logs warnings for skipped items so the user can see why data was dropped.
        When *diag* is provided, populates it with detailed diagnostic info.
        """
        healer = get_healer()

        if diag is not None:
            diag.llm_raw_response_length = len(raw)

        # --- Layered cleanup strategies ---
        def _clean_level_0(text: str) -> str:
            """Basic cleanup: strip markdown fences."""
            c = re.sub(r"```(?:json)?", "", text).strip()
            return c.strip("`").strip()

        def _clean_level_1(text: str) -> str:
            """Extract JSON array via regex."""
            match = re.search(r"\[.*\]", text, re.DOTALL)
            return match.group() if match else text

        def _clean_level_2(text: str) -> str:
            """Aggressive: fix common LLM JSON mistakes."""
            c = _clean_level_1(text)
            # Fix trailing commas before ] or }
            c = re.sub(r",\s*([}\]])", r"\1", c)
            # Fix single-quoted JSON keys/values (only structural quotes, not apostrophes)
            # Matches patterns like {'key': 'value'} but not contractions like "don't"
            c = re.sub(r"(?<=[\[{,:])\s*'", ' "', c)
            c = re.sub(r"'\s*(?=[,\]}:])", '"', c)
            # Remove control characters
            c = re.sub(r"[\x00-\x1f\x7f]", " ", c)
            return c

        def _clean_level_3(text: str) -> str:
            """Nuclear: extract individual JSON objects and rebuild array."""
            objects = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
            if objects:
                return "[" + ",".join(objects) + "]"
            return "[]"

        cleaners = [_clean_level_0, _clean_level_1, _clean_level_2, _clean_level_3]

        data = None
        last_error = None
        used_strategy = None

        for level, cleaner in enumerate(cleaners):
            try:
                cleaned = cleaner(raw)
                data = json.loads(cleaned)
                if level > 0:
                    used_strategy = f"json_cleanup_level_{level}"
                    log.info("JSON parsed at cleanup level %d for %s", level, source_file)
                if diag is not None:
                    diag.json_cleanup_level = level
                break
            except (json.JSONDecodeError, Exception) as e:
                last_error = e
                continue

        if data is None:
            err_msg = (
                f"All 4 JSON cleanup strategies failed "
                f"(response length: {len(raw)} chars, "
                f"last error: {last_error})"
            )
            log.warning("All JSON cleanup strategies failed for %s (response length: %d chars)",
                        source_file, len(raw))
            healer.record_failure(
                "json_parse",
                last_error or ValueError("All JSON cleanup strategies failed"),
                context={"source_file": source_file, "raw_length": len(raw)},
            )
            if diag is not None:
                diag.json_parse_ok = False
                diag.json_parse_error = err_msg
                # Include a snippet of the raw response to help debug
                snippet = raw[:300].replace("\n", " ")
                diag.warnings.append(f"LLM raw response preview: {snippet!r}")
            return []

        # Record success (with strategy if recovery was needed)
        healer.record_success(
            "json_parse",
            context={"source_file": source_file, "atom_count": len(data) if isinstance(data, list) else 0},
            recovery_strategy=used_strategy,
        )
        if used_strategy and last_error:
            healer.learn_strategy("json_parse", last_error, used_strategy, succeeded=True)

        if not isinstance(data, list):
            log.warning("LLM returned non-array JSON for %s (got %s)", source_file, type(data).__name__)
            if diag is not None:
                diag.json_parse_ok = False
                diag.json_parse_error = (
                    f"LLM returned {type(data).__name__} instead of a JSON array"
                )
            return []

        if diag is not None:
            diag.json_parse_ok = True
            diag.items_in_json = len(data)

        atoms: list[TechnicalAtom] = []
        skipped = 0
        for item in data:
            if not isinstance(item, dict):
                skipped += 1
                if diag is not None:
                    diag.skipped_reasons.append(
                        f"Non-dict item ({type(item).__name__}): {str(item)[:80]}"
                    )
                continue
            item.setdefault("source_file", source_file)
            try:
                atoms.append(TechnicalAtom(**item))
            except (TypeError, Exception) as exc:
                # Self-healing: try coercing fields to expected types
                if isinstance(exc, (TypeError, ValueError)):
                    try:
                        coerced = _coerce_atom_fields(item)
                        atoms.append(TechnicalAtom(**coerced))
                        healer.record_success(
                            "atom_construction",
                            context={"source_file": source_file},
                            recovery_strategy="coerce_types",
                        )
                        healer.learn_strategy(
                            "atom_construction", exc, "coerce_types", succeeded=True,
                        )
                        continue
                    except Exception:
                        healer.record_failure(
                            "atom_construction", exc,
                            context={"source_file": source_file},
                        )
                        healer.learn_strategy(
                            "atom_construction", exc, "coerce_types", succeeded=False,
                        )
                skipped += 1
                if diag is not None:
                    diag.skipped_reasons.append(
                        f"{type(exc).__name__}: {exc} — keys: {list(item.keys())}"
                    )
                log.debug("Skipped malformed atom in %s: %s — %s", source_file, exc, item)
        if skipped:
            log.warning("Skipped %d malformed item(s) from %s", skipped, source_file)

        if diag is not None:
            diag.atoms_constructed = len(atoms)
            diag.atoms_skipped = skipped
        return atoms

    @staticmethod
    def _normalise(atoms: list[TechnicalAtom]) -> list[TechnicalAtom]:
        """Normalise category names to canonical taxonomy values."""
        for atom in atoms:
            atom.category = normalise_category(atom.category).value
        return atoms
