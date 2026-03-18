"""Tests for chemistry.py — Chemistry Instruction Designer."""

from __future__ import annotations

import unittest

from engine import TechnicalAtom, TaxonomyCategory
from chemistry import (
    KitZone,
    SensoryType,
    SensoryCheckpoint,
    ActionResultStep,
    TroubleshootingEntry,
    build_action_result_steps,
    build_chemistry_instructions,
    embed_safety_in_action,
    generate_troubleshooting,
    infer_sensory_checkpoints,
    is_chemistry_taxonomy,
    organise_by_zones,
    zone_for_category,
)


def _atom(
    category: str = "Procedural Steps",
    parameter: str = "Step 1",
    value: str = "Mix chemicals",
    source: str = "test.txt",
    **kw,
) -> TechnicalAtom:
    return TechnicalAtom(
        category=category, parameter=parameter, value=value,
        source_file=source, **kw,
    )


# ---------------------------------------------------------------------------
# Zone mapping
# ---------------------------------------------------------------------------

class TestZoneMapping(unittest.TestCase):
    def test_tools_map_to_zone_a(self):
        self.assertEqual(
            zone_for_category(TaxonomyCategory.TOOLS_EQUIPMENT.value),
            KitZone.ZONE_A_SETUP,
        )

    def test_materials_map_to_zone_b(self):
        self.assertEqual(
            zone_for_category(TaxonomyCategory.MATERIALS.value),
            KitZone.ZONE_B_REAGENTS,
        )

    def test_procedural_maps_to_zone_c(self):
        self.assertEqual(
            zone_for_category(TaxonomyCategory.PROCEDURAL_STEPS.value),
            KitZone.ZONE_C_REACTION,
        )

    def test_logistics_maps_to_zone_d(self):
        self.assertEqual(
            zone_for_category(TaxonomyCategory.LOGISTICS_STORAGE.value),
            KitZone.ZONE_D_WASTE,
        )

    def test_unknown_defaults_to_zone_c(self):
        self.assertEqual(
            zone_for_category("Not A Real Category"),
            KitZone.ZONE_C_REACTION,
        )


class TestOrganiseByZones(unittest.TestCase):
    def test_atoms_sorted_into_zones(self):
        taxonomy = {
            "2. Tools & Equipment": [_atom(category="Tools & Equipment", parameter="Beaker")],
            "1. Materials": [_atom(category="Materials", parameter="NaOH")],
            "12. Procedural Steps": [_atom(parameter="Mix")],
        }
        zones = organise_by_zones(taxonomy)
        self.assertEqual(len(zones[KitZone.ZONE_A_SETUP]), 1)
        self.assertEqual(len(zones[KitZone.ZONE_B_REAGENTS]), 1)
        self.assertEqual(len(zones[KitZone.ZONE_C_REACTION]), 1)

    def test_empty_taxonomy_gives_empty_zones(self):
        zones = organise_by_zones({})
        for z in KitZone:
            self.assertEqual(zones[z], [])


# ---------------------------------------------------------------------------
# Sensory checkpoints
# ---------------------------------------------------------------------------

class TestSensoryCheckpoints(unittest.TestCase):
    def test_colour_triggers_visual(self):
        cps = infer_sensory_checkpoints("The liquid turns blue")
        self.assertTrue(any(c.sensory_type == SensoryType.VISUAL for c in cps))

    def test_fizz_triggers_auditory(self):
        cps = infer_sensory_checkpoints("A slight fizzing sound")
        self.assertTrue(any(c.sensory_type == SensoryType.AUDITORY for c in cps))

    def test_warm_triggers_tactile(self):
        cps = infer_sensory_checkpoints("The flask becomes warm")
        self.assertTrue(any(c.sensory_type == SensoryType.TACTILE for c in cps))

    def test_odour_triggers_olfactory(self):
        cps = infer_sensory_checkpoints("A pungent odour is released")
        self.assertTrue(any(c.sensory_type == SensoryType.OLFACTORY for c in cps))

    def test_no_keywords_returns_empty(self):
        cps = infer_sensory_checkpoints("Measure 10ml of water")
        self.assertEqual(cps, [])

    def test_no_duplicate_sensory_types(self):
        cps = infer_sensory_checkpoints(
            "The colour turns blue and it becomes transparent"
        )
        visual_count = sum(1 for c in cps if c.sensory_type == SensoryType.VISUAL)
        self.assertEqual(visual_count, 1)


# ---------------------------------------------------------------------------
# Action-Result steps
# ---------------------------------------------------------------------------

class TestActionResultSteps(unittest.TestCase):
    def test_basic_step_creation(self):
        atoms = [_atom(parameter="Add NaOH to water", value="Water swirls but stays clear")]
        steps = build_action_result_steps(atoms)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].step_number, 1)
        self.assertEqual(steps[0].action, "Add NaOH to water")
        self.assertEqual(steps[0].expected_result, "Water swirls but stays clear")

    def test_hazardous_detection(self):
        atoms = [_atom(parameter="Add concentrated acid", value="Exothermic reaction")]
        steps = build_action_result_steps(atoms)
        self.assertTrue(steps[0].is_hazardous)

    def test_non_hazardous_step(self):
        atoms = [_atom(parameter="Measure water", value="10ml in graduated cylinder")]
        steps = build_action_result_steps(atoms)
        self.assertFalse(steps[0].is_hazardous)

    def test_safe_pause_detection(self):
        atoms = [_atom(parameter="Let stand", value="Allow to cool to room temperature")]
        steps = build_action_result_steps(atoms)
        self.assertTrue(steps[0].is_safe_pause_point)

    def test_sensory_checkpoints_inferred(self):
        atoms = [_atom(parameter="Heat mixture", value="Solution turns clear")]
        steps = build_action_result_steps(atoms)
        self.assertTrue(len(steps[0].sensory_checkpoints) > 0)

    def test_step_numbering(self):
        atoms = [_atom(parameter=f"Step {i}") for i in range(3)]
        steps = build_action_result_steps(atoms)
        self.assertEqual([s.step_number for s in steps], [1, 2, 3])


# ---------------------------------------------------------------------------
# Safety embedding
# ---------------------------------------------------------------------------

class TestSafetyEmbedding(unittest.TestCase):
    def test_ppe_prepended(self):
        safety_atoms = [
            _atom(category="Safety & Compliance", parameter="PPE", value="Wear gloves"),
        ]
        result = embed_safety_in_action("Pick up the flask", safety_atoms)
        self.assertIn("nitrile gloves", result)
        self.assertIn("pick up the flask", result)

    def test_no_ppe_no_change(self):
        result = embed_safety_in_action("Measure 10ml", [])
        self.assertEqual(result, "Measure 10ml")

    def test_already_mentions_ppe(self):
        safety_atoms = [
            _atom(category="Safety & Compliance", parameter="PPE", value="Wear gloves"),
        ]
        result = embed_safety_in_action("While wearing gloves, pour", safety_atoms)
        self.assertEqual(result, "While wearing gloves, pour")


# ---------------------------------------------------------------------------
# Troubleshooting
# ---------------------------------------------------------------------------

class TestTroubleshooting(unittest.TestCase):
    def test_hazardous_step_generates_entry(self):
        steps = [ActionResultStep(
            step_number=1, action="Add acid", expected_result="Fizzing",
            is_hazardous=True,
        )]
        entries = generate_troubleshooting(steps)
        self.assertTrue(len(entries) > 0)
        self.assertTrue(any(e.severity == "critical" for e in entries))

    def test_troubleshooting_text_generates_entry(self):
        steps = [ActionResultStep(
            step_number=1, action="Mix", expected_result="Clear",
            troubleshooting="Check pH if cloudy",
        )]
        entries = generate_troubleshooting(steps)
        self.assertTrue(any("Check pH" in e.probable_cause for e in entries))

    def test_no_issues_no_entries(self):
        steps = [ActionResultStep(
            step_number=1, action="Pour water", expected_result="Water in beaker",
        )]
        entries = generate_troubleshooting(steps)
        self.assertEqual(entries, [])


# ---------------------------------------------------------------------------
# Chemistry detection
# ---------------------------------------------------------------------------

class TestChemistryDetection(unittest.TestCase):
    def test_chemistry_taxonomy_detected(self):
        taxonomy = {
            "1. Materials": [
                _atom(category="Materials", parameter="Reagent A", value="Hydrochloric acid 1M"),
                _atom(category="Materials", parameter="Solvent", value="Distilled water"),
                _atom(category="Materials", parameter="Buffer", value="pH 7 phosphate buffer"),
            ],
        }
        self.assertTrue(is_chemistry_taxonomy(taxonomy))

    def test_non_chemistry_taxonomy(self):
        taxonomy = {
            "1. Materials": [
                _atom(category="Materials", parameter="Steel type", value="304 stainless"),
                _atom(category="Materials", parameter="Width", value="25mm"),
            ],
        }
        self.assertFalse(is_chemistry_taxonomy(taxonomy))


# ---------------------------------------------------------------------------
# Full build
# ---------------------------------------------------------------------------

class TestBuildChemistryInstructions(unittest.TestCase):
    def test_full_build(self):
        taxonomy = {
            "2. Tools & Equipment": [
                _atom(category="Tools & Equipment", parameter="Beaker", value="500ml glass"),
            ],
            "10. Safety & Compliance": [
                _atom(category="Safety & Compliance", parameter="PPE", value="Wear nitrile gloves"),
            ],
            "1. Materials": [
                _atom(category="Materials", parameter="NaOH", value="0.1M solution"),
            ],
            "12. Procedural Steps": [
                _atom(parameter="Add NaOH to water", value="Solution turns cloudy then clear"),
                _atom(parameter="Let stand for 5 minutes", value="Allow to settle at room temperature"),
            ],
        }
        result = build_chemistry_instructions(taxonomy, "Test Procedure")
        self.assertEqual(result.title, "Test Procedure")
        self.assertEqual(len(result.steps), 2)
        # Safety should be embedded in actions
        self.assertIn("nitrile gloves", result.steps[0].action)
        # Second step should be a safe pause point
        self.assertTrue(result.steps[1].is_safe_pause_point)
        # Zones should be populated
        self.assertTrue(len(result.zones[KitZone.ZONE_A_SETUP]) > 0)
        self.assertTrue(len(result.zones[KitZone.ZONE_B_REAGENTS]) > 0)

    def test_empty_taxonomy(self):
        result = build_chemistry_instructions({})
        self.assertEqual(result.steps, [])
        self.assertEqual(result.troubleshooting, [])


if __name__ == "__main__":
    unittest.main()
