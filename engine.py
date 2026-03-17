"""
engine.py  –  The Operational Architect
Schema-First Extraction Brain
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# 20-Point Taxonomy
# ──────────────────────────────────────────────

TAXONOMY_SECTIONS: List[str] = [
    "1. Purpose/Success",
    "2. Hazards/Stop Conditions",
    "3. Workflow Map",
    "4. Materials",
    "5. Hardware",
    "6. Incoming Inspection",
    "7. Storage",
    "8. Build",
    "9. Integrity Checks",
    "10. Activation",
    "11. Preflight",
    "12. Live Procedure",
    "13. Monitoring",
    "14. Troubleshooting",
    "15. Shutdown",
    "16. Post-Run",
    "17. QC Release",
    "18. Maintenance",
    "19. Gap Audit",
    "20. Glossary",
]

HOISTABLE_SECTIONS = {"4. Materials", "5. Hardware"}

# Section aliases for fuzzy matching (lowercase keyword → canonical section)

_ALIASES: Dict[str, str] = {
    "purpose": "1. Purpose/Success",
    "success": "1. Purpose/Success",
    "goal": "1. Purpose/Success",
    "objective": "1. Purpose/Success",
    "hazard": "2. Hazards/Stop Conditions",
    "danger": "2. Hazards/Stop Conditions",
    "warning": "2. Hazards/Stop Conditions",
    "stop": "2. Hazards/Stop Conditions",
    "abort": "2. Hazards/Stop Conditions",
    "workflow": "3. Workflow Map",
    "process": "3. Workflow Map",
    "flow": "3. Workflow Map",
    "material": "4. Materials",
    "reagent": "4. Materials",
    "chemical": "4. Materials",
    "hardware": "5. Hardware",
    "equipment": "5. Hardware",
    "tool": "5. Hardware",
    "instrument": "5. Hardware",
    "inspection": "6. Incoming Inspection",
    "receiving": "6. Incoming Inspection",
    "incoming": "6. Incoming Inspection",
    "storage": "7. Storage",
    "store": "7. Storage",
    "shelf": "7. Storage",
    "build": "8. Build",
    "assembly": "8. Build",
    "fabrication": "8. Build",
    "integrity": "9. Integrity Checks",
    "leak": "9. Integrity Checks",
    "pressure": "9. Integrity Checks",
    "activation": "10. Activation",
    "startup": "10. Activation",
    "init": "10. Activation",
    "preflight": "11. Preflight",
    "pre-run": "11. Preflight",
    "checklist": "11. Preflight",
    "live": "12. Live Procedure",
    "procedure": "12. Live Procedure",
    "run": "12. Live Procedure",
    "operation": "12. Live Procedure",
    "monitor": "13. Monitoring",
    "log": "13. Monitoring",
    "record": "13. Monitoring",
    "troubleshoot": "14. Troubleshooting",
    "fault": "14. Troubleshooting",
    "error": "14. Troubleshooting",
    "shutdown": "15. Shutdown",
    "stop condition": "15. Shutdown",
    "post": "16. Post-Run",
    "cleanup": "16. Post-Run",
    "qc": "17. QC Release",
    "quality": "17. QC Release",
    "release": "17. QC Release",
    "maintenance": "18. Maintenance",
    "service": "18. Maintenance",
    "calibration": "18. Maintenance",
    "gap": "19. Gap Audit",
    "missing": "19. Gap Audit",
    "glossary": "20. Glossary",
    "definition": "20. Glossary",
    "term": "20. Glossary",
}

# ──────────────────────────────────────────────
# Data Model
# ──────────────────────────────────────────────


class TechnicalAtom(BaseModel):
    """Atomic unit of extracted technical knowledge."""

    category: str = Field(
        ...,
        description="One of the 20 taxonomy sections (e.g. '5. Hardware')",
    )
    parameter: str = Field(
        ...,
        description="The specific parameter, item, or step name",
    )
    value: str = Field(
        ...,
        description="The exact value, specification, measurement, or description",
    )
    source_file: str = Field(
        ...,
        description="Originating file name",
    )
    is_prerequisite: bool = Field(
        default=False,
        description="True if this atom must be satisfied before any subsequent steps",
    )
    hoisted: bool = Field(
        default=False,
        description="True if this atom was hoisted from a procedural section into Materials/Hardware",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "category": "5. Hardware",
                "parameter": "Tube Plug Material",
                "value": "316L Stainless Steel wool",
                "source_file": "assembly_notes.txt",
                "is_prerequisite": False,
                "hoisted": True,
            }
        }


# ──────────────────────────────────────────────
# Extraction System Prompt
# ──────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a Forensic Extraction Agent embedded inside an industrial Master Production Record system.

HARD RULES:

1. You are STRICTLY PROHIBITED from summarizing, paraphrasing, or omitting any technical data.
2. Every temperature, pressure, measurement, part number, material grade, safety warning, and \
   procedural step must be extracted as a discrete TechnicalAtom object.
3. If information is missing for a mandatory section, leave that section empty – DO NOT fabricate data.
4. Prioritize dependency order: any prerequisite item must have is_prerequisite=true.

THE 20 MANDATORY TAXONOMY SECTIONS:

1. Purpose/Success        11. Preflight
2. Hazards/Stop Conditions  12. Live Procedure
3. Workflow Map           13. Monitoring
4. Materials              14. Troubleshooting
5. Hardware               15. Shutdown
6. Incoming Inspection    16. Post-Run
7. Storage                17. QC Release
8. Build                  18. Maintenance
9. Integrity Checks       19. Gap Audit
10. Activation            20. Glossary

HOISTING RULE (MANDATORY):
If a tool, material, reagent, or equipment item is mentioned ANYWHERE inside a procedural \
step or description, you MUST create TWO atoms:
(a) One in the procedural section (e.g. "8. Build") describing the step.
(b) One ADDITIONAL atom in "4. Materials" or "5. Hardware" listing that item, \
with hoisted=true.

OUTPUT FORMAT:
Return ONLY a valid JSON array of TechnicalAtom objects.
Do NOT wrap in markdown code fences. Do NOT include any explanation text.
Each object MUST contain: category, parameter, value, source_file, is_prerequisite, hoisted.

EXAMPLE (truncated):
[
{
"category": "5. Hardware",
"parameter": "Tube Plug Material",
"value": "316L Stainless Steel wool",
"source_file": "notes.txt",
"is_prerequisite": false,
"hoisted": true
},
{
"category": "8. Build",
"parameter": "Tube Plugging Step",
"value": "Use 316L Stainless Steel wool to plug the tube before pressurizing the manifold",
"source_file": "notes.txt",
"is_prerequisite": true,
"hoisted": false
}
]
"""

# ──────────────────────────────────────────────
# Brain
# ──────────────────────────────────────────────


class OperationalBrain:
    """Orchestrates multi-file extraction and taxonomy organization."""

    def __init__(self, provider: str, api_key: str, model: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        if provider == "Anthropic":
            self.model = model or "claude-sonnet-4-20250514"
        else:
            self.model = model or "gpt-4o"

    # ── Public API ─────────────────────────────
    def extract_atoms(self, text: str, source_file: str) -> List[TechnicalAtom]:
        """Extract TechnicalAtoms from raw text."""
        user_prompt = (
            f"SOURCE FILE: {source_file}\n\n"
            f"DOCUMENT CONTENT:\n{text}\n\n"
            "Extract every piece of technical data into TechnicalAtom JSON objects. "
            "Remember the HOISTING RULE. Return only the JSON array."
        )
        raw = self._call_llm(user_prompt)
        atoms = self._parse_atoms(raw, source_file)
        return atoms

    def organize_by_taxonomy(
        self, atoms: List[TechnicalAtom]
    ) -> Dict[str, List[TechnicalAtom]]:
        """Bin atoms into the 20-section taxonomy dict."""
        result: Dict[str, List[TechnicalAtom]] = {s: [] for s in TAXONOMY_SECTIONS}
        for atom in atoms:
            canonical = self._normalize_category(atom.category)
            atom.category = canonical
            if canonical in result:
                result[canonical].append(atom)
            # silently drop if still no match (shouldn't happen post-normalize)
        return result

    def get_gap_report(
        self, taxonomy: Dict[str, List[TechnicalAtom]]
    ) -> List[str]:
        """Return list of sections with zero atoms (excluding Gap Audit itself)."""
        return [
            section
            for section in TAXONOMY_SECTIONS
            if section != "19. Gap Audit" and len(taxonomy.get(section, [])) == 0
        ]

    # ── LLM Call ──────────────────────────────
    def _call_llm(self, user_prompt: str) -> str:
        if self.provider == "Anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=8000,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        else:  # OpenAI
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=8000,
            )
            return response.choices[0].message.content or ""

    # ── Parsing & Normalization ────────────────
    def _parse_atoms(self, raw: str, source_file: str) -> List[TechnicalAtom]:
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = raw.strip("`").strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        atoms: List[TechnicalAtom] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            item["source_file"] = source_file  # enforce provenance
            try:
                atom = TechnicalAtom(**item)
                atoms.append(atom)
            except Exception:
                # Build best-effort atom from partial data
                try:
                    atom = TechnicalAtom(
                        category=str(item.get("category", "20. Glossary")),
                        parameter=str(item.get("parameter", "Unknown")),
                        value=str(item.get("value", str(item))),
                        source_file=source_file,
                        is_prerequisite=bool(item.get("is_prerequisite", False)),
                        hoisted=bool(item.get("hoisted", False)),
                    )
                    atoms.append(atom)
                except Exception:
                    continue
        return atoms

    def _normalize_category(self, category: str) -> str:
        """Map a raw LLM category string to a canonical taxonomy section."""
        cat = category.strip()

        # Exact match first
        if cat in TAXONOMY_SECTIONS:
            return cat

        # Number-prefix match (e.g. "4." or "4 " → "4. Materials")
        num_match = re.match(r"^(\d{1,2})[\.\s]", cat)
        if num_match:
            num = num_match.group(1)
            for section in TAXONOMY_SECTIONS:
                if section.startswith(f"{num}."):
                    return section

        cat_lower = cat.lower()

        # Alias lookup
        for keyword, canonical in _ALIASES.items():
            if keyword in cat_lower:
                return canonical

        # Substring match against section labels
        for section in TAXONOMY_SECTIONS:
            label = section.split(". ", 1)[1].lower()
            if label in cat_lower or cat_lower in label:
                return section

        # Default: dump to Glossary so nothing is lost
        return "20. Glossary"
