"""Tests for extraction diagnostics."""

from __future__ import annotations

import json

import pytest

from engine import (
    ExtractionDiagnostics,
    FileExtractionDiag,
    OperationalBrain,
    TechnicalAtom,
    UsageStats,
    _extraction_cache,
    clear_extraction_cache,
    content_hash,
    deduplicate_atoms,
    detect_duplicate_documents,
    get_cached_extraction,
    set_cached_extraction,
)


# ---------------------------------------------------------------------------
# FileExtractionDiag unit tests
# ---------------------------------------------------------------------------


class TestFileExtractionDiag:
    def test_no_problems_when_all_ok(self):
        d = FileExtractionDiag(
            filename="test.txt",
            file_read_ok=True,
            text_length=500,
            llm_call_ok=True,
            json_parse_ok=True,
            items_in_json=3,
            atoms_constructed=3,
            atoms_skipped=0,
            atoms_after_normalise=3,
        )
        assert not d.has_problems()

    def test_problem_when_file_read_fails(self):
        d = FileExtractionDiag(filename="bad.pdf", file_read_ok=False, file_read_error="corrupt")
        assert d.has_problems()

    def test_problem_when_llm_call_fails(self):
        d = FileExtractionDiag(filename="test.txt", llm_call_ok=False, llm_call_error="401")
        assert d.has_problems()

    def test_problem_when_json_parse_fails(self):
        d = FileExtractionDiag(filename="test.txt", json_parse_ok=False, json_parse_error="bad json")
        assert d.has_problems()

    def test_problem_when_atoms_skipped(self):
        d = FileExtractionDiag(
            filename="test.txt",
            atoms_constructed=2,
            atoms_skipped=3,
        )
        assert d.has_problems()

    def test_problem_when_json_items_but_zero_atoms(self):
        d = FileExtractionDiag(
            filename="test.txt",
            json_parse_ok=True,
            items_in_json=5,
            atoms_constructed=0,
        )
        assert d.has_problems()

    def test_problem_when_text_but_zero_atoms(self):
        d = FileExtractionDiag(
            filename="test.txt",
            text_length=1000,
            atoms_after_normalise=0,
        )
        assert d.has_problems()

    def test_summary_lines_file_read_error(self):
        d = FileExtractionDiag(filename="bad.pdf", file_read_ok=False, file_read_error="corrupt PDF")
        lines = d.summary_lines()
        assert any("File read failed" in l for l in lines)
        assert any("corrupt PDF" in l for l in lines)
        # Should stop early — no LLM or JSON lines
        assert not any("LLM" in l for l in lines)

    def test_summary_lines_llm_error(self):
        d = FileExtractionDiag(
            filename="test.txt",
            text_length=500,
            llm_call_ok=False,
            llm_call_error="AuthenticationError: invalid key",
        )
        lines = d.summary_lines()
        assert any("Text extracted" in l for l in lines)
        assert any("LLM call failed" in l for l in lines)
        # Should stop early — no JSON lines
        assert not any("JSON" in l for l in lines)

    def test_summary_lines_json_error(self):
        d = FileExtractionDiag(
            filename="test.txt",
            text_length=500,
            llm_raw_response_length=200,
            json_parse_ok=False,
            json_parse_error="All 4 strategies failed",
        )
        lines = d.summary_lines()
        assert any("LLM response" in l for l in lines)
        assert any("JSON parsing failed" in l for l in lines)

    def test_summary_lines_full_success(self):
        d = FileExtractionDiag(
            filename="test.txt",
            text_length=1000,
            llm_raw_response_length=500,
            json_parse_ok=True,
            json_cleanup_level=0,
            items_in_json=5,
            atoms_constructed=5,
            atoms_skipped=0,
            atoms_after_normalise=5,
        )
        lines = d.summary_lines()
        assert any("1,000" in l for l in lines)  # text length formatted
        assert any("5" in l and "items" in l.lower() for l in lines)
        assert any("5" in l and "constructed" in l.lower() for l in lines)

    def test_summary_lines_with_skipped_reasons(self):
        d = FileExtractionDiag(
            filename="test.txt",
            json_parse_ok=True,
            items_in_json=5,
            atoms_constructed=3,
            atoms_skipped=2,
            skipped_reasons=["TypeError: missing field", "ValueError: bad confidence"],
        )
        lines = d.summary_lines()
        assert any("skipped" in l.lower() and "2" in l for l in lines)
        assert any("TypeError" in l for l in lines)


# ---------------------------------------------------------------------------
# ExtractionDiagnostics aggregate tests
# ---------------------------------------------------------------------------


class TestExtractionDiagnostics:
    def test_total_atoms(self):
        diags = ExtractionDiagnostics()
        diags.add(FileExtractionDiag(filename="a.txt", atoms_after_normalise=5))
        diags.add(FileExtractionDiag(filename="b.txt", atoms_after_normalise=3))
        assert diags.total_atoms() == 8

    def test_total_skipped(self):
        diags = ExtractionDiagnostics()
        diags.add(FileExtractionDiag(filename="a.txt", atoms_skipped=2))
        diags.add(FileExtractionDiag(filename="b.txt", atoms_skipped=1))
        assert diags.total_skipped() == 3

    def test_files_with_problems(self):
        diags = ExtractionDiagnostics()
        ok = FileExtractionDiag(filename="ok.txt", atoms_after_normalise=5, text_length=100)
        ok.atoms_constructed = 5
        ok.items_in_json = 5
        bad = FileExtractionDiag(filename="bad.txt", llm_call_ok=False)
        diags.add(ok)
        diags.add(bad)
        problems = diags.files_with_problems()
        assert len(problems) == 1
        assert problems[0].filename == "bad.txt"

    def test_has_any_problems(self):
        diags = ExtractionDiagnostics()
        diags.add(FileExtractionDiag(filename="bad.txt", file_read_ok=False))
        assert diags.has_any_problems()

    def test_no_problems_when_all_ok(self):
        diags = ExtractionDiagnostics()
        ok = FileExtractionDiag(
            filename="ok.txt",
            text_length=100,
            atoms_after_normalise=5,
            atoms_constructed=5,
            items_in_json=5,
        )
        diags.add(ok)
        assert not diags.has_any_problems()


# ---------------------------------------------------------------------------
# Integration: _parse_atoms populates diagnostics
# ---------------------------------------------------------------------------


class TestParseAtomsWithDiag:
    """Verify that _parse_atoms populates the FileExtractionDiag correctly."""

    def _make_valid_json(self, atoms_data: list[dict]) -> str:
        return json.dumps(atoms_data)

    def test_successful_parse_populates_diag(self):
        raw = self._make_valid_json([
            {
                "category": "Materials",
                "parameter": "Steel grade",
                "value": "304L",
                "source_file": "test.txt",
            }
        ])
        diag = FileExtractionDiag(filename="test.txt")
        atoms = OperationalBrain._parse_atoms(raw, "test.txt", diag)
        assert len(atoms) == 1
        assert diag.json_parse_ok is True
        assert diag.items_in_json == 1
        assert diag.atoms_constructed == 1
        assert diag.atoms_skipped == 0
        assert diag.llm_raw_response_length == len(raw)
        assert diag.json_cleanup_level == 0

    def test_bad_json_populates_diag(self):
        # Text with no brackets: the nuclear cleanup (level 3) produces "[]",
        # which parses as valid empty JSON. So json_parse_ok is True but 0 items.
        raw = "This is not JSON at all, just plain text with no brackets."
        diag = FileExtractionDiag(filename="test.txt")
        atoms = OperationalBrain._parse_atoms(raw, "test.txt", diag)
        assert len(atoms) == 0
        assert diag.json_parse_ok is True
        assert diag.items_in_json == 0
        assert diag.json_cleanup_level == 3  # required nuclear cleanup

    def test_malformed_atoms_tracked_in_diag(self):
        raw = self._make_valid_json([
            {
                "category": "Materials",
                "parameter": "Steel grade",
                "value": "304L",
                "source_file": "test.txt",
            },
            {"bad_key": "no category or parameter"},
        ])
        diag = FileExtractionDiag(filename="test.txt")
        atoms = OperationalBrain._parse_atoms(raw, "test.txt", diag)
        assert len(atoms) == 1
        assert diag.items_in_json == 2
        assert diag.atoms_constructed == 1
        assert diag.atoms_skipped == 1
        assert len(diag.skipped_reasons) == 1

    def test_json_with_markdown_fences_parsed(self):
        inner = json.dumps([{
            "category": "Materials",
            "parameter": "test",
            "value": "val",
            "source_file": "f.txt",
        }])
        raw = f"```json\n{inner}\n```"
        diag = FileExtractionDiag(filename="f.txt")
        atoms = OperationalBrain._parse_atoms(raw, "f.txt", diag)
        assert len(atoms) == 1
        assert diag.json_parse_ok is True

    def test_non_array_json_populates_diag(self):
        raw = json.dumps({"not": "an array"})
        diag = FileExtractionDiag(filename="test.txt")
        atoms = OperationalBrain._parse_atoms(raw, "test.txt", diag)
        assert len(atoms) == 0
        assert diag.json_parse_ok is False
        assert "array" in diag.json_parse_error.lower()

    def test_diag_none_still_works(self):
        """Passing diag=None should not break anything (backward compat)."""
        raw = self._make_valid_json([{
            "category": "Materials",
            "parameter": "test",
            "value": "val",
            "source_file": "f.txt",
        }])
        atoms = OperationalBrain._parse_atoms(raw, "f.txt", None)
        assert len(atoms) == 1

    def test_empty_response_populates_diag(self):
        raw = ""
        diag = FileExtractionDiag(filename="test.txt")
        atoms = OperationalBrain._parse_atoms(raw, "test.txt", diag)
        assert len(atoms) == 0
        assert diag.llm_raw_response_length == 0


# ---------------------------------------------------------------------------
# clear_extraction_cache tests
# ---------------------------------------------------------------------------


class TestClearExtractionCache:
    def setup_method(self):
        _extraction_cache.clear()

    def test_clear_empty_cache_returns_zero(self):
        assert clear_extraction_cache() == 0

    def test_clear_populated_cache_returns_count(self):
        set_cached_extraction("txt", "a.pdf", "anthropic", "model-1", "[{}]")
        set_cached_extraction("txt2", "b.pdf", "openai", "model-2", "[{}]")
        assert clear_extraction_cache() == 2

    def test_cache_is_empty_after_clear(self):
        set_cached_extraction("txt", "a.pdf", "anthropic", "model-1", "[{}]")
        clear_extraction_cache()
        assert get_cached_extraction("txt", "a.pdf", "anthropic", "model-1") is None

    def test_clear_removes_bad_cached_response(self):
        """A bad LLM response cached should be gone after clear."""
        set_cached_extraction("doc", "f.pdf", "anthropic", "m", "NOT JSON")
        assert get_cached_extraction("doc", "f.pdf", "anthropic", "m") == "NOT JSON"
        clear_extraction_cache()
        assert get_cached_extraction("doc", "f.pdf", "anthropic", "m") is None

    def test_cached_bad_response_poisons_retries(self):
        """Demonstrates the bug: a bad LLM response gets cached and
        every subsequent extraction returns the same bad result."""
        bad_response = "Sorry, I cannot extract specs from this document."
        set_cached_extraction("doc", "f.txt", "anthropic", "m", bad_response)
        cached = get_cached_extraction("doc", "f.txt", "anthropic", "m")
        assert cached == bad_response
        clear_extraction_cache()
        assert get_cached_extraction("doc", "f.txt", "anthropic", "m") is None


# ---------------------------------------------------------------------------
# UsageStats.reset tests
# ---------------------------------------------------------------------------


class TestUsageStatsReset:
    def test_reset_clears_all_fields(self):
        stats = UsageStats()
        stats.record("gpt-4o", 500, 200)
        stats.record("gpt-4o", 300, 100)
        stats.record_cache_hit()
        assert stats.calls == 2
        assert stats.cache_hits == 1
        assert stats.total_input_tokens > 0

        stats.reset()

        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_cost_usd == 0.0
        assert stats.calls == 0
        assert stats.cache_hits == 0

    def test_reset_allows_reuse(self):
        stats = UsageStats()
        stats.record("gpt-4o-mini", 1000, 500)
        stats.reset()
        stats.record("gpt-4o-mini", 200, 100)
        s = stats.summary()
        assert s["calls"] == 1
        assert s["input_tokens"] == 200
        assert s["output_tokens"] == 100


# ---------------------------------------------------------------------------
# Document deduplication tests
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_identical_content_same_hash(self):
        assert content_hash("hello world") == content_hash("hello world")

    def test_whitespace_normalised(self):
        assert content_hash("hello   world") == content_hash("hello world")
        assert content_hash("hello\n\nworld") == content_hash("hello world")
        assert content_hash("  hello world  ") == content_hash("hello world")

    def test_different_content_different_hash(self):
        assert content_hash("hello") != content_hash("world")


class TestDetectDuplicateDocuments:
    def test_no_duplicates(self):
        texts = {"a.txt": "content A", "b.txt": "content B"}
        groups = detect_duplicate_documents(texts)
        assert len(groups) == 0

    def test_two_identical_files(self):
        texts = {"a.txt": "same content", "b.txt": "same content"}
        groups = detect_duplicate_documents(texts)
        assert len(groups) == 1
        assert set(groups[0].filenames) == {"a.txt", "b.txt"}
        assert groups[0].keeper in ("a.txt", "b.txt")
        assert len(groups[0].duplicates) == 1

    def test_whitespace_differences_detected(self):
        texts = {"a.txt": "hello  world", "b.txt": "hello world"}
        groups = detect_duplicate_documents(texts)
        assert len(groups) == 1

    def test_three_identical_files(self):
        texts = {"a.txt": "same", "b.txt": "same", "c.txt": "same"}
        groups = detect_duplicate_documents(texts)
        assert len(groups) == 1
        assert len(groups[0].filenames) == 3
        assert len(groups[0].duplicates) == 2

    def test_two_separate_dup_groups(self):
        texts = {
            "a.txt": "content A",
            "b.txt": "content A",
            "c.txt": "content B",
            "d.txt": "content B",
        }
        groups = detect_duplicate_documents(texts)
        assert len(groups) == 2

    def test_single_file_no_group(self):
        texts = {"only.txt": "just one"}
        groups = detect_duplicate_documents(texts)
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# Atom deduplication tests
# ---------------------------------------------------------------------------


def _make_atom(category="Materials", parameter="Steel grade", value="304L",
               source="a.txt", confidence=1.0):
    return TechnicalAtom(
        category=category, parameter=parameter, value=value,
        source_file=source, confidence=confidence,
    )


class TestDeduplicateAtoms:
    def test_no_duplicates_unchanged(self):
        atoms = [
            _make_atom(parameter="Steel grade", value="304L"),
            _make_atom(parameter="Bolt size", value="M10"),
        ]
        result = deduplicate_atoms(atoms)
        assert len(result.kept) == 2
        assert result.removed_count == 0
        assert result.duplicate_groups == 0

    def test_exact_duplicates_merged(self):
        atoms = [
            _make_atom(source="a.txt", confidence=0.9),
            _make_atom(source="b.txt", confidence=1.0),
        ]
        result = deduplicate_atoms(atoms)
        assert len(result.kept) == 1
        assert result.removed_count == 1
        assert result.duplicate_groups == 1
        # Higher confidence wins
        assert result.kept[0].confidence == 1.0

    def test_different_values_not_merged(self):
        atoms = [
            _make_atom(value="304L", source="a.txt"),
            _make_atom(value="316L", source="b.txt"),
        ]
        result = deduplicate_atoms(atoms)
        assert len(result.kept) == 2
        assert result.removed_count == 0

    def test_case_insensitive_dedup(self):
        atoms = [
            _make_atom(value="304L", source="a.txt"),
            _make_atom(value="304l", source="b.txt"),
        ]
        result = deduplicate_atoms(atoms)
        assert len(result.kept) == 1
        assert result.removed_count == 1

    def test_merged_atom_has_source_note(self):
        atoms = [
            _make_atom(source="a.txt", confidence=1.0),
            _make_atom(source="b.txt", confidence=0.8),
        ]
        result = deduplicate_atoms(atoms)
        assert result.kept[0].source_file == "a.txt"
        assert "b.txt" in (result.kept[0].notes or "")

    def test_three_duplicates(self):
        atoms = [
            _make_atom(source="a.txt", confidence=0.7),
            _make_atom(source="b.txt", confidence=0.9),
            _make_atom(source="c.txt", confidence=0.8),
        ]
        result = deduplicate_atoms(atoms)
        assert len(result.kept) == 1
        assert result.removed_count == 2
        assert result.kept[0].confidence == 0.9  # highest wins

    def test_same_source_duplicates_merged(self):
        atoms = [
            _make_atom(source="a.txt"),
            _make_atom(source="a.txt"),
        ]
        result = deduplicate_atoms(atoms)
        assert len(result.kept) == 1
        assert result.removed_count == 1
