"""Tests for layman_v3.py — Layman Procedural Document v3 pipeline."""

from __future__ import annotations

import json
import os
import unittest

from layman_v3 import (
    ValidationReport,
    _parse_llm_json,
    _build_system_prompt,
    _collect_all_step_ids,
    _collect_all_material_ids,
    _collect_all_zone_ids,
    fix_v3_document,
    load_v3_schema,
    validate_v3_document,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_doc(**overrides) -> dict:
    """Return a minimal valid v3 document for testing."""
    doc = {
        "meta": {
            "title": "Test Procedure",
            "objective": "Test the extraction pipeline.",
            "estimatedTotalTime": "1 hour",
        },
        "acquire": {
            "materials": [
                {
                    "id": "mat-001",
                    "commonName": "Distilled Water",
                    "visualId": {"appearance": "Clear liquid in a plastic bottle"},
                    "quantity": "500ml",
                    "sourceType": "lab_supply",
                    "zone": "zone-reagents",
                    "substitutes": {"best": {"name": "Deionized water", "why": "Same purity"}},
                    "storage": {"direction": "Keep at room temperature", "why": "Stable"},
                    "usedIn": [{"stepId": "step-001", "purpose": "Solvent"}],
                },
            ],
        },
        "stage": {
            "zones": [
                {
                    "id": "zone-reagents",
                    "name": "Reagents Zone",
                    "description": "Where chemicals are stored before use.",
                    "materialIds": ["mat-001"],
                },
            ],
        },
        "execute": {
            "phases": [
                {
                    "id": "phase-001",
                    "name": "Preparation",
                    "summary": "Get everything ready.",
                    "dangerLevel": "safe",
                    "ppeForPhase": ["safety goggles", "nitrile gloves"],
                    "steps": [
                        {
                            "id": "step-001",
                            "action": "Pour 500ml of distilled water into the beaker.",
                            "sensoryCheckpoint": {
                                "right": {
                                    "description": "Clear liquid fills the beaker to the 500ml mark.",
                                    "senses": ["sight"],
                                },
                                "wrong": {
                                    "description": "Liquid is cloudy or discolored.",
                                    "consequence": "Contaminated water will ruin the experiment.",
                                    "recovery": "Discard and use a fresh bottle of distilled water.",
                                },
                            },
                        },
                    ],
                },
            ],
        },
        "verify": {
            "tests": [
                {
                    "id": "verify-001",
                    "name": "pH Check",
                    "method": "Dip pH strip into solution.",
                    "pass": {"description": "Strip shows pH 7 (green)"},
                    "fail": {
                        "description": "Strip shows acidic or basic reading.",
                        "recovery": "Add buffer solution and retest.",
                    },
                },
            ],
        },
        "shutdown": {
            "steps": [
                {
                    "id": "sd-001",
                    "action": "Turn off the hot plate.",
                    "observation": "Red indicator light goes off.",
                },
            ],
        },
        "emergencyShutdown": {
            "triggerConditions": ["Fire", "Chemical spill", "Injury"],
            "steps": [
                {
                    "id": "esd-001",
                    "verb": "CLOSE",
                    "target": "gas valve",
                    "physicalLocation": "Left side of fume hood",
                },
                {
                    "id": "esd-002",
                    "verb": "EXIT",
                    "target": "laboratory",
                },
            ],
        },
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

class TestSchemaLoading(unittest.TestCase):
    def test_schema_file_exists(self):
        self.assertTrue(os.path.isfile(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "layman_v3_schema.json")
        ))

    def test_schema_loads(self):
        schema = load_v3_schema()
        self.assertIsInstance(schema, dict)

    def test_schema_has_required_keys(self):
        schema = load_v3_schema()
        props = schema.get("properties", {})
        for key in ["meta", "acquire", "stage", "execute", "verify", "shutdown", "emergencyShutdown"]:
            self.assertIn(key, props, f"Schema missing property: {key}")

    def test_schema_has_required_field(self):
        schema = load_v3_schema()
        self.assertIn("required", schema)
        for key in ["meta", "acquire", "stage", "execute", "verify", "shutdown", "emergencyShutdown"]:
            self.assertIn(key, schema["required"])


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

class TestJsonParsing(unittest.TestCase):
    def test_clean_json(self):
        raw = '{"meta": {"title": "Test"}}'
        result = _parse_llm_json(raw)
        self.assertEqual(result["meta"]["title"], "Test")

    def test_markdown_fenced_json(self):
        raw = '```json\n{"meta": {"title": "Test"}}\n```'
        result = _parse_llm_json(raw)
        self.assertEqual(result["meta"]["title"], "Test")

    def test_json_with_preamble(self):
        raw = 'Here is the output:\n\n{"meta": {"title": "Test"}}'
        result = _parse_llm_json(raw)
        self.assertEqual(result["meta"]["title"], "Test")

    def test_trailing_comma_cleanup(self):
        raw = '{"meta": {"title": "Test",}}'
        result = _parse_llm_json(raw)
        self.assertEqual(result["meta"]["title"], "Test")

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            _parse_llm_json("This is not JSON at all.")


# ---------------------------------------------------------------------------
# ID collection helpers
# ---------------------------------------------------------------------------

class TestIdCollectors(unittest.TestCase):
    def test_collect_step_ids(self):
        doc = _minimal_doc()
        ids = _collect_all_step_ids(doc)
        self.assertEqual(ids, {"step-001"})

    def test_collect_material_ids(self):
        doc = _minimal_doc()
        ids = _collect_all_material_ids(doc)
        self.assertEqual(ids, {"mat-001"})

    def test_collect_zone_ids(self):
        doc = _minimal_doc()
        ids = _collect_all_zone_ids(doc)
        self.assertEqual(ids, {"zone-reagents"})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):
    def test_valid_document_passes(self):
        doc = _minimal_doc()
        report = validate_v3_document(doc)
        self.assertTrue(report.valid)
        self.assertEqual(report.errors, [])

    def test_missing_top_level_section(self):
        doc = _minimal_doc()
        del doc["emergencyShutdown"]
        report = validate_v3_document(doc)
        self.assertFalse(report.valid)
        self.assertTrue(any("emergencyShutdown" in e for e in report.errors))

    def test_duplicate_step_id(self):
        doc = _minimal_doc()
        # Add a second step with same ID
        doc["execute"]["phases"][0]["steps"].append({
            "id": "step-001",
            "action": "Duplicate step.",
            "sensoryCheckpoint": {
                "right": {"description": "OK"},
                "wrong": {"description": "Bad", "consequence": "Fail", "recovery": "Redo"},
            },
        })
        report = validate_v3_document(doc)
        self.assertFalse(report.valid)
        self.assertTrue(any("Duplicate step ID" in e for e in report.errors))

    def test_duplicate_material_id(self):
        doc = _minimal_doc()
        doc["acquire"]["materials"].append({
            "id": "mat-001",
            "commonName": "Duplicate",
            "visualId": {"appearance": "Clear"},
            "quantity": "1L",
            "sourceType": "lab_supply",
            "zone": "zone-reagents",
            "substitutes": {"best": {"name": "X", "why": "Y"}},
            "storage": {"direction": "Cool", "why": "Stable"},
            "usedIn": [],
        })
        report = validate_v3_document(doc)
        self.assertFalse(report.valid)

    def test_dangling_zone_ref_warns(self):
        doc = _minimal_doc()
        doc["acquire"]["materials"][0]["zone"] = "nonexistent-zone"
        report = validate_v3_document(doc)
        self.assertTrue(any("nonexistent-zone" in w for w in report.warnings))

    def test_dangling_step_ref_warns(self):
        doc = _minimal_doc()
        doc["acquire"]["materials"][0]["usedIn"] = [
            {"stepId": "nonexistent-step", "purpose": "Test"},
        ]
        report = validate_v3_document(doc)
        self.assertTrue(any("nonexistent-step" in w for w in report.warnings))

    def test_missing_sensory_checkpoint_warns(self):
        doc = _minimal_doc()
        del doc["execute"]["phases"][0]["steps"][0]["sensoryCheckpoint"]
        report = validate_v3_document(doc)
        self.assertTrue(any("sensoryCheckpoint" in w for w in report.warnings))

    def test_missing_wrong_consequence_warns(self):
        doc = _minimal_doc()
        del doc["execute"]["phases"][0]["steps"][0]["sensoryCheckpoint"]["wrong"]["consequence"]
        report = validate_v3_document(doc)
        self.assertTrue(any("consequence" in w for w in report.warnings))

    def test_glossary_duplicate_warns(self):
        doc = _minimal_doc()
        step = doc["execute"]["phases"][0]["steps"][0]
        step["glossaryTerms"] = [
            {"term": "Distillation", "definition": "Boiling and condensing"},
        ]
        # Add second phase with duplicate term
        doc["execute"]["phases"].append({
            "id": "phase-002",
            "name": "Reaction",
            "summary": "React.",
            "dangerLevel": "caution",
            "steps": [{
                "id": "step-002",
                "action": "Heat the mixture.",
                "sensoryCheckpoint": {
                    "right": {"description": "Boiling"},
                    "wrong": {"description": "No bubbles", "consequence": "Fail", "recovery": "Reheat"},
                },
                "glossaryTerms": [
                    {"term": "Distillation", "definition": "Same thing again"},
                ],
            }],
        })
        report = validate_v3_document(doc)
        self.assertTrue(any("distillation" in w.lower() for w in report.warnings))

    def test_emergency_shutdown_verb_validation(self):
        doc = _minimal_doc()
        doc["emergencyShutdown"]["steps"][0]["verb"] = "close valve"
        report = validate_v3_document(doc)
        self.assertTrue(any("single word" in w for w in report.warnings))

    def test_emergency_shutdown_verb_uppercase_validation(self):
        doc = _minimal_doc()
        doc["emergencyShutdown"]["steps"][0]["verb"] = "close"
        report = validate_v3_document(doc)
        self.assertTrue(any("UPPERCASE" in w for w in report.warnings))

    def test_missing_danger_level_warns(self):
        doc = _minimal_doc()
        del doc["execute"]["phases"][0]["dangerLevel"]
        report = validate_v3_document(doc)
        self.assertTrue(any("dangerLevel" in w for w in report.warnings))

    def test_stats_populated(self):
        doc = _minimal_doc()
        report = validate_v3_document(doc)
        self.assertEqual(report.stats["materials"], 1)
        self.assertEqual(report.stats["phases"], 1)
        self.assertEqual(report.stats["steps"], 1)
        self.assertEqual(report.stats["emergency_steps"], 2)


# ---------------------------------------------------------------------------
# Auto-repair
# ---------------------------------------------------------------------------

class TestAutoRepair(unittest.TestCase):
    def test_assigns_missing_ids(self):
        doc = _minimal_doc()
        doc["execute"]["phases"][0]["steps"][0]["id"] = ""
        report = validate_v3_document(doc)
        fixed = fix_v3_document(doc, report)
        step_id = fixed["execute"]["phases"][0]["steps"][0]["id"]
        self.assertTrue(step_id.startswith("step-"))

    def test_sets_completed_false(self):
        doc = _minimal_doc()
        report = validate_v3_document(doc)
        fixed = fix_v3_document(doc, report)
        step = fixed["execute"]["phases"][0]["steps"][0]
        self.assertFalse(step["completed"])
        sd = fixed["shutdown"]["steps"][0]
        self.assertFalse(sd["completed"])

    def test_deduplicates_glossary(self):
        doc = _minimal_doc()
        doc["execute"]["phases"][0]["steps"][0]["glossaryTerms"] = [
            {"term": "pH", "definition": "Measure of acidity"},
        ]
        doc["execute"]["phases"].append({
            "id": "phase-002",
            "name": "Phase 2",
            "summary": "Next phase.",
            "dangerLevel": "safe",
            "steps": [{
                "id": "step-002",
                "action": "Test pH.",
                "sensoryCheckpoint": {
                    "right": {"description": "Green"},
                    "wrong": {"description": "Red", "consequence": "Bad", "recovery": "Redo"},
                },
                "glossaryTerms": [
                    {"term": "pH", "definition": "Duplicate definition"},
                ],
            }],
        })
        report = validate_v3_document(doc)
        fixed = fix_v3_document(doc, report)
        # First step keeps the term
        self.assertEqual(len(fixed["execute"]["phases"][0]["steps"][0]["glossaryTerms"]), 1)
        # Second step should have it removed
        self.assertEqual(len(fixed["execute"]["phases"][1]["steps"][0]["glossaryTerms"]), 0)

    def test_uppercases_esd_verbs(self):
        doc = _minimal_doc()
        doc["emergencyShutdown"]["steps"][0]["verb"] = "close"
        report = validate_v3_document(doc)
        fixed = fix_v3_document(doc, report)
        self.assertEqual(fixed["emergencyShutdown"]["steps"][0]["verb"], "CLOSE")

    def test_removes_dangling_cross_refs(self):
        doc = _minimal_doc()
        doc["acquire"]["materials"][0]["usedIn"] = [
            {"stepId": "nonexistent", "purpose": "Ghost"},
            {"stepId": "step-001", "purpose": "Real"},
        ]
        report = validate_v3_document(doc)
        fixed = fix_v3_document(doc, report)
        used_in = fixed["acquire"]["materials"][0]["usedIn"]
        self.assertEqual(len(used_in), 1)
        self.assertEqual(used_in[0]["stepId"], "step-001")

    def test_flags_missing_fields(self):
        doc = _minimal_doc()
        report = validate_v3_document(doc)
        fixed = fix_v3_document(doc, report)
        step = fixed["execute"]["phases"][0]["steps"][0]
        self.assertEqual(step.get("_flag_estimatedTime"), "not_in_source")
        self.assertTrue(len(report.flags) > 0)

    def test_creates_missing_top_level_sections(self):
        doc = _minimal_doc()
        del doc["shutdown"]
        report = ValidationReport()
        report.errors.append("Missing required top-level section: 'shutdown'")
        fixed = fix_v3_document(doc, report)
        self.assertIn("shutdown", fixed)
        self.assertIsInstance(fixed["shutdown"]["steps"], list)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

class TestPromptConstruction(unittest.TestCase):
    def test_system_prompt_contains_schema(self):
        schema = load_v3_schema()
        prompt = _build_system_prompt(schema)
        self.assertIn('"meta"', prompt)
        self.assertIn('"acquire"', prompt)
        self.assertIn('"execute"', prompt)

    def test_system_prompt_contains_processing_rules(self):
        schema = load_v3_schema()
        prompt = _build_system_prompt(schema)
        self.assertIn("Chronological ordering", prompt)
        self.assertIn("Atomic steps", prompt)
        self.assertIn("Direction of Direction", prompt)

    def test_system_prompt_contains_output_constraints(self):
        schema = load_v3_schema()
        prompt = _build_system_prompt(schema)
        self.assertIn("Return ONLY valid JSON", prompt)


if __name__ == "__main__":
    unittest.main()
