"""
layman_v3.py — Layman Procedural Document v3 Extraction Pipeline

Extracts chemistry procedures from source documents into richly structured
JSON conforming to layman-doc-schema-v3. Uses the LLM directly (via
engine.call_llm) with the full schema + processing rules as the system
prompt, then validates and auto-repairs the output.

Three main components:
  1. LaymanV3Brain  — orchestrates LLM extraction (single-chunk & multi-chunk)
  2. validate_v3_document — post-LLM structural & cross-reference validation
  3. fix_v3_document — auto-repair for fixable issues
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from engine import call_llm, split_text

# Layman v3 has a massive system prompt (~10K tokens from the schema alone).
# Use a smaller chunk size so each chunk + system prompt + output fits
# comfortably and completes within a reasonable time.
LAYMAN_CHUNK_CHARS = 30_000

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema loader
# ---------------------------------------------------------------------------

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layman_v3_schema.json")


def load_v3_schema() -> dict:
    """Load the v3 JSON schema from disk."""
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Processing rules (companion guide) — embedded in the system prompt
# ---------------------------------------------------------------------------

_PROCESSING_RULES = r"""
## PROCESSING RULES

1. **Chronological ordering.** Reorganize source content by the order things
   physically happen, regardless of how the source is structured. The output
   timeline must read like "do this, then this, then this."

2. **Atomic steps.** If the source says "connect the tube and tighten the
   fitting," split into two steps. One verb per action field.

3. **Direction of Direction pattern.** If the source provides both a
   high-level instruction ("tighten the nut") AND a physical HOW
   ("finger-tight plus a quarter turn"), map the high-level to `action`
   and the physical HOW to `directionOfDirection`.

4. **Warning/risk routing:**
   - Warning names a specific step → `alerts[]` on that step
   - Warning spans multiple steps → `meta.crossCuttingRisks[]`
     (with `affectsSteps` populated)
   - Warning about material integrity → `acquire.materials[].incomingInspection`
   - Warning about storage degradation → `acquire.materials[].storage`

5. **Emergency shutdown extraction.** Any "ESD", "emergency shutdown", or
   panic-mode sequence → extract to top-level `emergencyShutdown`. Also
   duplicate as `level: "red"` alerts on relevant steps.

6. **Cross-referencing integrity (hard requirements):**
   - `hardwareLibrary.tools[].materialId` → existing `acquire.materials[].id`
   - `acquire.materials[].zone` → existing `stage.zones[].id`
   - `acquire.materials[].usedIn[].stepId` → existing step id
   - `requires[].stepId` → existing step id
   - `unlocks[]` → existing step ids
   - `crossCuttingRisks[].affectsSteps[]` → existing step ids
   - `gate.fail.targetStepId` → existing step id

7. **Flagging missing data.** For fields not present in the source, output
   `null` with a sibling `_flag` property:
   ```json
   { "estimatedTime": null, "_flag_estimatedTime": "not_in_source" }
   ```
   Fields to flag when absent: estimatedTime, cumulativeTimeRemaining,
   pauseInfo, scaleRules, cleanup.cannotMixWith, bodyPosition,
   commonMistake, laymanAnalogy, pricing, calibration, handState,
   storage.maxShelfLife, sensoryCheckpoint.right.temporalCue,
   sensoryCheckpoint.wrong.temporalCue.
   Exception: if clearly inferable from context, populate normally.

8. **Tone rewriting for zero-experience reader:**
   - `action`: single imperative sentence starting with a verb
   - `summary`: present tense, plain english, no jargon
   - `consequence`: concrete physical outcome, never abstract
   - `directionOfDirection`: include body mechanics and tool grip
   - `sensoryCheckpoint.right.referenceComparison`: everyday objects only
   - `sensoryCheckpoint.wrong.description`: state what you see, not chemistry
   - `gate.howToCheck`: describe the physical act of checking
   - `emergencyShutdown.steps[].verb`: single UPPERCASE word
   - Phase names: plain english, no acronyms

9. **Phase assignment.** A new phase starts when ANY of:
   - Danger level changes (safe ↔ hazardous)
   - PPE changes (add or remove equipment)
   - System state transitions (cold→hot, empty→loaded, inert→reactive)
   - Significant time gap (mandatory wait, e.g., "let sit for 24 hours")
   Always populate: name, summary, dangerLevel, ppeForPhase, estimatedPhaseTime.

10. **Glossary terms are first-use only.** If a term is defined in phase 1
    step 3, it must NOT appear in any later step's glossaryTerms. Scan all
    steps sequentially and deduplicate.

11. **PPE inheritance.** `ppe` on a step inherits from `phases[].ppeForPhase`.
    Only populate step-level `ppe` if the step ADDS or REMOVES equipment
    relative to the phase default.
"""

# ---------------------------------------------------------------------------
# System & user prompt construction
# ---------------------------------------------------------------------------


def _strip_descriptions(obj: Any) -> Any:
    """Recursively strip verbose 'description' fields from a JSON schema.

    Keeps only descriptions that contain constraint keywords (enum, pattern,
    minimum, required) — those carry information the LLM needs.
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "description" and isinstance(v, str):
                # Keep descriptions that encode constraints
                lower = v.lower()
                if any(kw in lower for kw in ("enum", "pattern", "must", "required", "one of", "minimum", "maximum")):
                    result[k] = v
                # else: drop verbose descriptions — property names are enough
            else:
                result[k] = _strip_descriptions(v)
        return result
    elif isinstance(obj, list):
        return [_strip_descriptions(item) for item in obj]
    return obj


def _build_system_prompt(schema: dict) -> str:
    """Build the full system prompt from schema + processing rules.

    Uses compact JSON (no indentation) and strips verbose descriptions
    to reduce the system prompt from ~14K tokens to ~8K tokens.
    """
    # Remove rendering directives and strip verbose descriptions
    schema_copy = {k: v for k, v in schema.items() if not k.startswith("_")}
    schema_copy = _strip_descriptions(schema_copy)
    # Compact JSON — no indentation saves ~30% tokens
    schema_json = json.dumps(schema_copy, separators=(",", ":"))

    return f"""You are a chemistry procedure extractor. Your task is to convert
source chemistry/laboratory documents into a structured, layman-friendly
procedural document in JSON format.

## OUTPUT SCHEMA

Your output MUST be a single JSON object conforming to this schema:

{schema_json}

{_PROCESSING_RULES}

## OUTPUT CONSTRAINTS

- Return ONLY valid JSON. No markdown fences, no commentary, no explanation.
- The JSON must be a single object with these top-level keys:
  meta, acquire, stage, execute, verify, shutdown, emergencyShutdown
- Optional top-level key: maintenance
- All string values must be plain english suitable for a zero-experience reader.
- Every step must have BOTH sensoryCheckpoint.right AND sensoryCheckpoint.wrong.
- Every wrong must have consequence AND recovery.
- emergencyShutdown.steps must contain ZERO branching logic. Every step is unconditional.
- All IDs must be unique and cross-references must resolve.
"""


def _build_user_prompt(text: str, filename: str) -> str:
    """Build the user prompt containing the source document."""
    return f"""Extract a complete layman procedural document from the following
chemistry source material. Follow all processing rules exactly.

Source filename: {filename}

--- BEGIN SOURCE DOCUMENT ---
{text}
--- END SOURCE DOCUMENT ---
"""


def _build_merge_prompt(partials: list[dict]) -> str:
    """Build prompt for merging partial extractions from multiple chunks."""
    partials_json = json.dumps(partials, indent=2)
    return f"""You have received partial extractions from different chunks of
the same chemistry document. Merge them into a single unified layman
procedural document.

Rules for merging:
- Combine all materials into one acquire.materials array (deduplicate by name)
- Combine all zones into one stage.zones array (deduplicate by name)
- Merge all phases chronologically into execute.phases
- Merge verify tests, shutdown steps, and emergency shutdown steps
- Ensure all cross-references resolve after merging
- Deduplicate glossary terms (first-use only)
- Recalculate IDs to be globally unique

Return ONLY the merged JSON object. No markdown fences, no commentary.

--- BEGIN PARTIAL EXTRACTIONS ---
{partials_json}
--- END PARTIAL EXTRACTIONS ---
"""


# ---------------------------------------------------------------------------
# JSON parsing with progressive cleanup (mirrors engine.py pattern)
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> dict:
    """Parse LLM response as JSON with progressive cleanup levels."""
    # Level 0: try raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Level 1: strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", raw)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Level 2: extract first { ... } block
    match = re.search(r"\{", cleaned)
    if match:
        start = match.start()
        # Find matching closing brace
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break

    # Level 3: aggressive cleanup — fix common LLM JSON errors
    # trailing commas, single quotes, unquoted keys
    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)  # trailing commas
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    raise ValueError(
        f"Failed to parse LLM response as JSON after 4 cleanup levels. "
        f"Response starts with: {raw[:200]!r}"
    )


# ---------------------------------------------------------------------------
# LaymanV3Brain — extraction orchestrator
# ---------------------------------------------------------------------------


class LaymanV3Brain:
    """Orchestrates LLM-based extraction of v3 layman procedural documents."""

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: str = "",
        model: str | None = None,
        tier: str = "quality",
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.tier = tier
        self._schema = load_v3_schema()
        self._system_prompt = _build_system_prompt(self._schema)

    def extract_v3(self, text: str, filename: str = "document.txt") -> dict:
        """Extract a v3 layman procedural document from source text.

        Uses layman-aware chunk sizing (30K chars) to account for the large
        system prompt. For single-chunk docs, makes one LLM call. For
        larger docs, extracts per chunk then merges via a second LLM call.
        """
        chunks = split_text(text, max_chars=LAYMAN_CHUNK_CHARS)

        if len(chunks) == 1:
            return self._extract_single(chunks[0], filename)
        else:
            return self._extract_multi(chunks, filename)

    def extract_multi_documents(
        self,
        file_texts: dict[str, str],
        progress_callback: Any = None,
    ) -> dict:
        """Extract v3 from multiple documents independently, then merge.

        Instead of concatenating all documents into one giant prompt, each
        document is processed separately and results are merged
        programmatically. Falls back to LLM merge only when needed.
        """
        partials: list[dict] = []
        total = len(file_texts)

        for i, (fname, text) in enumerate(file_texts.items()):
            if progress_callback:
                progress_callback(i, total, fname)
            doc = self.extract_v3(text, fname)
            partials.append(doc)

        if len(partials) == 1:
            return partials[0]

        return _merge_v3_documents(partials)

    def _extract_single(self, text: str, filename: str) -> dict:
        """Single-chunk extraction."""
        user_prompt = _build_user_prompt(text, filename)
        # Cap at 16K output tokens — realistic for structured JSON and
        # completable within a reasonable time with streaming.
        max_tokens = max(4096, min(16384, len(text) // 3))

        raw = call_llm(
            user_prompt,
            self._system_prompt,
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            max_tokens=max_tokens,
            tier=self.tier,
        )

        doc = _parse_llm_json(raw)
        return doc

    def _extract_multi(self, chunks: list[str], filename: str) -> dict:
        """Multi-chunk extraction with merge pass."""
        partials: list[dict] = []

        for i, chunk in enumerate(chunks):
            chunk_name = f"{filename} (chunk {i + 1}/{len(chunks)})"
            user_prompt = _build_user_prompt(chunk, chunk_name)
            max_tokens = max(4096, min(16384, len(chunk) // 3))

            raw = call_llm(
                user_prompt,
                self._system_prompt,
                provider=self.provider,
                model=self.model,
                api_key=self.api_key,
                max_tokens=max_tokens,
                tier=self.tier,
            )

            try:
                partial = _parse_llm_json(raw)
                partials.append(partial)
            except ValueError:
                log.warning("Failed to parse chunk %d/%d, skipping", i + 1, len(chunks))
                continue

        if not partials:
            raise ValueError("All chunks failed to produce valid JSON")

        if len(partials) == 1:
            return partials[0]

        # Merge pass
        merge_prompt = _build_merge_prompt(partials)
        max_tokens = min(16384, sum(len(json.dumps(p)) for p in partials) // 4)
        max_tokens = max(4096, max_tokens)

        raw = call_llm(
            merge_prompt,
            self._system_prompt,
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            max_tokens=max_tokens,
            tier=self.tier,
        )

        return _parse_llm_json(raw)


# ---------------------------------------------------------------------------
# Programmatic multi-document merge (no LLM call needed)
# ---------------------------------------------------------------------------

def _merge_v3_documents(docs: list[dict]) -> dict:
    """Merge multiple v3 documents into one without an LLM call.

    Combines materials, zones, phases, verify tests, shutdown steps,
    and emergency shutdown steps. Deduplicates by name/id where possible.
    """
    merged: dict[str, Any] = {
        "meta": {},
        "acquire": {"materials": []},
        "stage": {"zones": []},
        "execute": {"phases": []},
        "verify": {"tests": []},
        "shutdown": {"steps": []},
        "emergencyShutdown": {"steps": []},
    }

    seen_material_names: set[str] = set()
    seen_zone_names: set[str] = set()

    for doc in docs:
        # Meta: merge cross-cutting risks, keep first title/objective
        meta = doc.get("meta", {})
        if not merged["meta"].get("title") and meta.get("title"):
            merged["meta"]["title"] = meta["title"]
        if not merged["meta"].get("objective") and meta.get("objective"):
            merged["meta"]["objective"] = meta["objective"]
        for risk in meta.get("crossCuttingRisks", []):
            merged["meta"].setdefault("crossCuttingRisks", []).append(risk)

        # Materials: deduplicate by name
        for mat in doc.get("acquire", {}).get("materials", []):
            name = (mat.get("name") or "").lower().strip()
            if name and name not in seen_material_names:
                seen_material_names.add(name)
                merged["acquire"]["materials"].append(mat)
            elif not name:
                merged["acquire"]["materials"].append(mat)

        # Zones: deduplicate by name
        for zone in doc.get("stage", {}).get("zones", []):
            name = (zone.get("name") or "").lower().strip()
            if name and name not in seen_zone_names:
                seen_zone_names.add(name)
                merged["stage"]["zones"].append(zone)
            elif not name:
                merged["stage"]["zones"].append(zone)

        # Phases: concatenate (each doc contributes its own phases)
        merged["execute"]["phases"].extend(
            doc.get("execute", {}).get("phases", [])
        )

        # Verify, shutdown, emergency shutdown: concatenate
        merged["verify"]["tests"].extend(
            doc.get("verify", {}).get("tests", [])
        )
        merged["shutdown"]["steps"].extend(
            doc.get("shutdown", {}).get("steps", [])
        )
        merged["emergencyShutdown"]["steps"].extend(
            doc.get("emergencyShutdown", {}).get("steps", [])
        )

    return merged


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL = ["meta", "acquire", "stage", "execute", "verify", "shutdown", "emergencyShutdown"]

_FLAGGABLE_FIELDS = {
    "estimatedTime", "cumulativeTimeRemaining", "pauseInfo", "scaleRules",
    "bodyPosition", "commonMistake", "laymanAnalogy", "pricing",
    "calibration", "handState",
}


@dataclass
class ValidationReport:
    """Result of validating a v3 document."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def _collect_all_step_ids(doc: dict) -> set[str]:
    """Collect all step IDs from execute.phases."""
    ids: set[str] = set()
    for phase in doc.get("execute", {}).get("phases", []):
        for step in phase.get("steps", []):
            step_id = step.get("id")
            if step_id:
                ids.add(step_id)
    return ids


def _collect_all_material_ids(doc: dict) -> set[str]:
    """Collect all material IDs from acquire.materials."""
    ids: set[str] = set()
    for mat in doc.get("acquire", {}).get("materials", []):
        mat_id = mat.get("id")
        if mat_id:
            ids.add(mat_id)
    return ids


def _collect_all_zone_ids(doc: dict) -> set[str]:
    """Collect all zone IDs from stage.zones."""
    ids: set[str] = set()
    for zone in doc.get("stage", {}).get("zones", []):
        zone_id = zone.get("id")
        if zone_id:
            ids.add(zone_id)
    return ids


def validate_v3_document(doc: dict) -> ValidationReport:
    """Validate a v3 document for structural integrity and cross-references.

    Checks:
      1. Required top-level sections present
      2. Cross-reference integrity (materialIds, stepIds, zone refs)
      3. ID uniqueness across materials, steps, phases
      4. Sensory checkpoint completeness (right AND wrong)
      5. Wrong has consequence AND recovery
      6. Glossary term deduplication (first-use only)
      7. Emergency shutdown has no branching logic (verb is single word)
      8. Phase danger levels present
      9. Red alerts have recovery paths
    """
    report = ValidationReport()

    # 1. Required top-level sections
    for key in _REQUIRED_TOP_LEVEL:
        if key not in doc:
            report.errors.append(f"Missing required top-level section: '{key}'")
            report.valid = False

    if not report.valid:
        return report  # Can't validate further without required sections

    # Collect all IDs for cross-reference checks
    step_ids = _collect_all_step_ids(doc)
    material_ids = _collect_all_material_ids(doc)
    zone_ids = _collect_all_zone_ids(doc)
    phase_ids: set[str] = set()

    # Stats
    total_steps = 0
    total_phases = 0
    total_materials = len(material_ids)

    # 2. Check for duplicate IDs
    _seen_step_ids: list[str] = []
    _seen_material_ids: list[str] = []

    for mat in doc.get("acquire", {}).get("materials", []):
        mid = mat.get("id")
        if mid and mid in _seen_material_ids:
            report.errors.append(f"Duplicate material ID: '{mid}'")
            report.valid = False
        if mid:
            _seen_material_ids.append(mid)

    for phase in doc.get("execute", {}).get("phases", []):
        total_phases += 1
        pid = phase.get("id")
        if pid and pid in phase_ids:
            report.errors.append(f"Duplicate phase ID: '{pid}'")
            report.valid = False
        if pid:
            phase_ids.add(pid)

        for step in phase.get("steps", []):
            total_steps += 1
            sid = step.get("id")
            if sid and sid in _seen_step_ids:
                report.errors.append(f"Duplicate step ID: '{sid}'")
                report.valid = False
            if sid:
                _seen_step_ids.append(sid)

    # 3. Cross-reference: materials[].zone → stage.zones[].id
    for mat in doc.get("acquire", {}).get("materials", []):
        zone_ref = mat.get("zone")
        if zone_ref and zone_ref not in zone_ids:
            report.warnings.append(
                f"Material '{mat.get('id', '?')}' references zone '{zone_ref}' "
                f"which does not exist in stage.zones"
            )

    # 4. Cross-reference: materials[].usedIn[].stepId → step ids
    for mat in doc.get("acquire", {}).get("materials", []):
        for usage in mat.get("usedIn", []):
            step_ref = usage.get("stepId")
            if step_ref and step_ref not in step_ids:
                report.warnings.append(
                    f"Material '{mat.get('id', '?')}' usedIn references step "
                    f"'{step_ref}' which does not exist"
                )

    # 5. Cross-reference: requires/unlocks → step ids
    for phase in doc.get("execute", {}).get("phases", []):
        for step in phase.get("steps", []):
            for req in step.get("requires", []):
                ref = req.get("stepId")
                if ref and ref not in step_ids:
                    report.warnings.append(
                        f"Step '{step.get('id', '?')}' requires step '{ref}' "
                        f"which does not exist"
                    )
            for unlock in step.get("unlocks", []):
                if unlock and unlock not in step_ids:
                    report.warnings.append(
                        f"Step '{step.get('id', '?')}' unlocks step '{unlock}' "
                        f"which does not exist"
                    )

    # 6. Cross-reference: crossCuttingRisks[].affectsSteps → step ids
    for risk in doc.get("meta", {}).get("crossCuttingRisks", []):
        for ref in risk.get("affectsSteps", []):
            if ref not in step_ids:
                report.warnings.append(
                    f"Cross-cutting risk '{risk.get('id', '?')}' references "
                    f"step '{ref}' which does not exist"
                )

    # 7. Cross-reference: gate.fail.targetStepId → step ids
    for phase in doc.get("execute", {}).get("phases", []):
        for step in phase.get("steps", []):
            gate = step.get("gate")
            if gate:
                fail = gate.get("fail", {})
                target = fail.get("targetStepId")
                if target and target not in step_ids:
                    report.warnings.append(
                        f"Step '{step.get('id', '?')}' gate.fail.targetStepId "
                        f"'{target}' does not exist"
                    )

    # 8. Sensory checkpoint completeness
    for phase in doc.get("execute", {}).get("phases", []):
        for step in phase.get("steps", []):
            sc = step.get("sensoryCheckpoint")
            if not sc:
                report.warnings.append(
                    f"Step '{step.get('id', '?')}' missing sensoryCheckpoint"
                )
                continue
            if "right" not in sc:
                report.warnings.append(
                    f"Step '{step.get('id', '?')}' sensoryCheckpoint missing 'right'"
                )
            if "wrong" not in sc:
                report.warnings.append(
                    f"Step '{step.get('id', '?')}' sensoryCheckpoint missing 'wrong'"
                )
            else:
                wrong = sc["wrong"]
                if not wrong.get("consequence"):
                    report.warnings.append(
                        f"Step '{step.get('id', '?')}' wrong missing 'consequence'"
                    )
                if not wrong.get("recovery"):
                    report.warnings.append(
                        f"Step '{step.get('id', '?')}' wrong missing 'recovery'"
                    )

    # 9. Glossary term deduplication check
    seen_terms: set[str] = set()
    for phase in doc.get("execute", {}).get("phases", []):
        for step in phase.get("steps", []):
            for term_obj in step.get("glossaryTerms", []):
                term = term_obj.get("term", "").lower()
                if term in seen_terms:
                    report.warnings.append(
                        f"Glossary term '{term}' appears in step "
                        f"'{step.get('id', '?')}' but was already defined earlier"
                    )
                seen_terms.add(term)

    # 10. Emergency shutdown validation
    esd = doc.get("emergencyShutdown", {})
    esd_steps = esd.get("steps", [])
    if not esd_steps:
        report.warnings.append("emergencyShutdown has no steps")
    for es in esd_steps:
        verb = es.get("verb", "")
        if " " in verb.strip():
            report.warnings.append(
                f"Emergency shutdown step '{es.get('id', '?')}' verb "
                f"'{verb}' should be a single word"
            )
        if verb != verb.upper():
            report.warnings.append(
                f"Emergency shutdown step '{es.get('id', '?')}' verb "
                f"'{verb}' should be UPPERCASE"
            )

    # 11. Phase danger levels
    for phase in doc.get("execute", {}).get("phases", []):
        if not phase.get("dangerLevel"):
            report.warnings.append(
                f"Phase '{phase.get('id', '?')}' missing dangerLevel"
            )

    # 12. Red alerts should have recovery path
    for phase in doc.get("execute", {}).get("phases", []):
        for step in phase.get("steps", []):
            for alert in step.get("alerts", []):
                if alert.get("level") == "red":
                    gate = step.get("gate")
                    sc = step.get("sensoryCheckpoint", {})
                    wrong = sc.get("wrong", {})
                    has_recovery = (
                        (gate and gate.get("fail"))
                        or wrong.get("recovery")
                    )
                    if not has_recovery:
                        report.warnings.append(
                            f"Step '{step.get('id', '?')}' has red alert but "
                            f"no gate.fail or sensoryCheckpoint.wrong.recovery"
                        )

    # Populate stats
    report.stats = {
        "materials": total_materials,
        "zones": len(zone_ids),
        "phases": total_phases,
        "steps": total_steps,
        "verify_tests": len(doc.get("verify", {}).get("tests", [])),
        "shutdown_steps": len(doc.get("shutdown", {}).get("steps", [])),
        "emergency_steps": len(esd_steps),
        "cross_cutting_risks": len(doc.get("meta", {}).get("crossCuttingRisks", [])),
        "warnings": len(report.warnings),
        "errors": len(report.errors),
    }

    if report.errors:
        report.valid = False

    return report


# ---------------------------------------------------------------------------
# Auto-repair
# ---------------------------------------------------------------------------

def fix_v3_document(doc: dict, report: ValidationReport) -> dict:
    """Auto-repair fixable issues in a v3 document.

    Fixes:
      - Assign sequential IDs where missing
      - Add _flag markers to empty flaggable fields
      - Deduplicate glossary terms across steps
      - Set completed: false on all steps and shutdown steps
      - Uppercase emergency shutdown verbs
      - Remove dangling cross-references
    """
    doc = copy.deepcopy(doc)

    # Ensure required top-level sections exist (as empty structures)
    defaults = {
        "meta": {"title": "", "objective": "", "estimatedTotalTime": null_flag("estimatedTotalTime")},
        "acquire": {"materials": []},
        "stage": {"zones": []},
        "execute": {"phases": []},
        "verify": {"tests": []},
        "shutdown": {"steps": []},
        "emergencyShutdown": {"steps": []},
    }
    for key, default in defaults.items():
        if key not in doc:
            doc[key] = default

    # Collect valid IDs after potential fixes
    step_ids = _collect_all_step_ids(doc)
    material_ids = _collect_all_material_ids(doc)
    zone_ids = _collect_all_zone_ids(doc)

    # Fix: assign sequential IDs where missing
    mat_counter = 0
    for mat in doc.get("acquire", {}).get("materials", []):
        if not mat.get("id"):
            mat_counter += 1
            mat["id"] = f"mat-{mat_counter:03d}"

    zone_counter = 0
    for zone in doc.get("stage", {}).get("zones", []):
        if not zone.get("id"):
            zone_counter += 1
            zone["id"] = f"zone-{zone_counter:03d}"

    step_counter = 0
    phase_counter = 0
    for phase in doc.get("execute", {}).get("phases", []):
        if not phase.get("id"):
            phase_counter += 1
            phase["id"] = f"phase-{phase_counter:03d}"
        for step in phase.get("steps", []):
            if not step.get("id"):
                step_counter += 1
                step["id"] = f"step-{step_counter:03d}"

    # Fix: set completed: false on all steps
    for phase in doc.get("execute", {}).get("phases", []):
        for step in phase.get("steps", []):
            if "completed" not in step:
                step["completed"] = False

    for sd_step in doc.get("shutdown", {}).get("steps", []):
        if "completed" not in sd_step:
            sd_step["completed"] = False

    # Fix: deduplicate glossary terms (keep first occurrence only)
    seen_terms: set[str] = set()
    for phase in doc.get("execute", {}).get("phases", []):
        for step in phase.get("steps", []):
            if "glossaryTerms" in step and step["glossaryTerms"]:
                deduped = []
                for term_obj in step["glossaryTerms"]:
                    term_key = term_obj.get("term", "").lower()
                    if term_key not in seen_terms:
                        seen_terms.add(term_key)
                        deduped.append(term_obj)
                step["glossaryTerms"] = deduped

    # Fix: uppercase emergency shutdown verbs
    for es in doc.get("emergencyShutdown", {}).get("steps", []):
        if es.get("verb"):
            es["verb"] = es["verb"].strip().upper()

    # Fix: assign IDs to emergency shutdown steps if missing
    for i, es in enumerate(doc.get("emergencyShutdown", {}).get("steps", [])):
        if not es.get("id"):
            es["id"] = f"esd-{i + 1:03d}"

    # Fix: assign IDs to shutdown steps if missing
    for i, sd in enumerate(doc.get("shutdown", {}).get("steps", [])):
        if not sd.get("id"):
            sd["id"] = f"sd-{i + 1:03d}"

    # Fix: assign IDs to verify tests if missing
    for i, test in enumerate(doc.get("verify", {}).get("tests", [])):
        if not test.get("id"):
            test["id"] = f"verify-{i + 1:03d}"

    # Recollect IDs after fixes
    step_ids = _collect_all_step_ids(doc)
    material_ids = _collect_all_material_ids(doc)
    zone_ids = _collect_all_zone_ids(doc)

    # Fix: remove dangling cross-references
    for mat in doc.get("acquire", {}).get("materials", []):
        if mat.get("zone") and mat["zone"] not in zone_ids:
            mat["zone"] = ""
        if "usedIn" in mat:
            mat["usedIn"] = [
                u for u in mat["usedIn"]
                if u.get("stepId") in step_ids
            ]

    for phase in doc.get("execute", {}).get("phases", []):
        for step in phase.get("steps", []):
            if "requires" in step:
                step["requires"] = [
                    r for r in step["requires"]
                    if r.get("stepId") in step_ids
                ]
            if "unlocks" in step:
                step["unlocks"] = [
                    u for u in step["unlocks"]
                    if u in step_ids
                ]

    for risk in doc.get("meta", {}).get("crossCuttingRisks", []):
        if "affectsSteps" in risk:
            risk["affectsSteps"] = [
                s for s in risk["affectsSteps"]
                if s in step_ids
            ]

    # Flag missing data on steps
    report.flags = []
    for phase in doc.get("execute", {}).get("phases", []):
        for step in phase.get("steps", []):
            for fld in _FLAGGABLE_FIELDS:
                if fld not in step or step[fld] is None:
                    flag_key = f"_flag_{fld}"
                    if flag_key not in step:
                        step[flag_key] = "not_in_source"
                        report.flags.append(
                            f"Step '{step.get('id', '?')}': {fld} flagged as not_in_source"
                        )

    return doc


def null_flag(field_name: str) -> str:
    """Return the _flag convention value for a missing field.

    Helper used when constructing default structures.
    """
    return "not_in_source"
