"""
app.py — 5-tab Streamlit UI for the Technical Taxonomy System.

Tabs:
  1. Extract    — Upload documents and run LLM extraction
  2. Resolve    — Detect & resolve cross-document conflicts
  3. MPR View   — Master Parameter Register grouped by taxonomy
  4. Gap Audit  — Highlight missing taxonomy sections post-resolution
  5. Export     — Download as Markdown or PDF
"""

from __future__ import annotations

import io
from typing import Any

import streamlit as st

from engine import OperationalBrain, TaxonomyCategory, TechnicalAtom, normalise_category
from resolver import (
    Conflict,
    apply_resolutions,
    detect_conflicts,
    resolve_conflict,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Technical Taxonomy System", layout="wide")

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "atoms": [],
    "conflicts": [],
    "resolved_atoms": [],
    "extracted_files": [],
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar — provider & model config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("LLM Configuration")
    provider = st.selectbox("Provider", ["anthropic", "openai"], index=0)
    api_key = st.text_input("API Key", type="password")
    model = st.text_input(
        "Model (optional)",
        placeholder="claude-sonnet-4-20250514 / gpt-4o",
    )
    model = model.strip() or None

# ---------------------------------------------------------------------------
# Tab definitions
# ---------------------------------------------------------------------------

tab_extract, tab_resolve, tab_mpr, tab_gap, tab_export = st.tabs(
    ["Extract", "Resolve", "MPR View", "Gap Audit", "Export"]
)

# ===========================  TAB 1 — EXTRACT  ============================

with tab_extract:
    st.subheader("Document Extraction")
    uploaded_files = st.file_uploader(
        "Upload one or more text / markdown / plain files",
        accept_multiple_files=True,
        type=["txt", "md", "text", "csv", "log"],
    )

    if st.button("Run Extraction", disabled=not api_key):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            brain = OperationalBrain(
                provider=provider, model=model, api_key=api_key
            )
            all_atoms: list[TechnicalAtom] = []
            progress = st.progress(0, text="Extracting…")
            for idx, uf in enumerate(uploaded_files):
                text = uf.read().decode("utf-8", errors="replace")
                with st.spinner(f"Extracting from {uf.name}…"):
                    atoms = brain.extract(uf.name, text)
                all_atoms.extend(atoms)
                progress.progress(
                    (idx + 1) / len(uploaded_files),
                    text=f"Done: {uf.name} ({len(atoms)} atoms)",
                )
            st.session_state["atoms"] = all_atoms
            st.session_state["extracted_files"] = [f.name for f in uploaded_files]
            st.session_state["conflicts"] = []
            st.session_state["resolved_atoms"] = []
            st.success(f"Extracted **{len(all_atoms)}** atoms from {len(uploaded_files)} file(s).")

    # Show extracted atoms
    if st.session_state["atoms"]:
        st.markdown(f"**{len(st.session_state['atoms'])} atoms** in memory.")
        with st.expander("View raw atoms"):
            for i, atom in enumerate(st.session_state["atoms"]):
                st.json(atom.model_dump(), expanded=False)

# ===========================  TAB 2 — RESOLVE  ============================

with tab_resolve:
    st.subheader("Conflict Resolution")

    atoms: list[TechnicalAtom] = st.session_state["atoms"]

    if st.button("Detect Conflicts"):
        if not atoms:
            st.warning("No atoms to analyse. Run extraction first.")
        else:
            conflicts = detect_conflicts(atoms)
            st.session_state["conflicts"] = conflicts
            if conflicts:
                st.warning(f"Found **{len(conflicts)}** conflict(s).")
            else:
                st.success("No conflicts detected!")
                st.session_state["resolved_atoms"] = list(atoms)

    conflicts: list[Conflict] = st.session_state["conflicts"]

    for idx, conflict in enumerate(conflicts):
        with st.expander(
            f"{'✅' if conflict.is_resolved else '⚠️'} {conflict.category} → {conflict.parameter}",
            expanded=not conflict.is_resolved,
        ):
            st.markdown(f"**Category:** {conflict.category}")
            st.markdown(f"**Parameter:** {conflict.parameter}")
            st.markdown("**Conflicting values:**")
            for a in conflict.atoms:
                st.markdown(
                    f"- `{a.value}` (confidence {a.confidence:.0%}) — *{a.source_file}*"
                )

            col1, col2 = st.columns(2)
            with col1:
                strategy = st.selectbox(
                    "Strategy",
                    ["keep_highest_confidence", "keep_first", "keep_last", "manual"],
                    key=f"strat_{idx}",
                )
            with col2:
                manual_val = st.text_input(
                    "Manual value (if manual)",
                    key=f"manual_{idx}",
                    disabled=strategy != "manual",
                )

            if st.button("Resolve", key=f"resolve_{idx}"):
                resolve_conflict(
                    conflict,
                    strategy=strategy,
                    manual_value=manual_val if strategy == "manual" else None,
                )
                st.success(f"Resolved → `{conflict.resolved_atom.value}`")  # type: ignore[union-attr]

    # Apply all resolutions
    if conflicts and st.button("Apply All Resolutions"):
        unresolved = [c for c in conflicts if not c.is_resolved]
        if unresolved:
            st.error(f"{len(unresolved)} conflict(s) still unresolved.")
        else:
            resolved = apply_resolutions(atoms, conflicts)
            st.session_state["resolved_atoms"] = resolved
            st.success(
                f"Applied resolutions. **{len(resolved)}** atoms in resolved set."
            )

# ===========================  TAB 3 — MPR VIEW  ==========================

with tab_mpr:
    st.subheader("Master Parameter Register")

    source = st.session_state["resolved_atoms"] or st.session_state["atoms"]
    if not source:
        st.info("No atoms yet. Run extraction first.")
    else:
        brain_view = OperationalBrain()
        grouped = brain_view.classify_by_taxonomy(source)
        for cat_name, cat_atoms in grouped.items():
            if not cat_atoms:
                continue
            with st.expander(f"{cat_name} ({len(cat_atoms)})", expanded=True):
                for atom in cat_atoms:
                    val = f"**{atom.parameter}**: {atom.value}"
                    if atom.unit:
                        val += f" {atom.unit}"
                    val += f"  *(source: {atom.source_file})*"
                    st.markdown(val)

# ===========================  TAB 4 — GAP AUDIT  =========================

with tab_gap:
    st.subheader("Gap Audit")
    st.caption("Runs against the resolved taxonomy (post-resolution state).")

    source_gap = st.session_state["resolved_atoms"] or st.session_state["atoms"]
    if not source_gap:
        st.info("No atoms yet. Run extraction first.")
    else:
        brain_gap = OperationalBrain()
        grouped_gap = brain_gap.classify_by_taxonomy(source_gap)
        covered = 0
        for cat in TaxonomyCategory:
            count = len(grouped_gap.get(cat.value, []))
            if count > 0:
                st.markdown(f"🟢 **{cat.value}** — {count} atom(s)")
                covered += 1
            else:
                st.markdown(f"🔴 **{cat.value}** — *no data*")

        st.divider()
        total = len(TaxonomyCategory)
        pct = covered / total * 100
        st.metric("Taxonomy Coverage", f"{covered}/{total} ({pct:.0f}%)")

# ===========================  TAB 5 — EXPORT  ============================


def _build_markdown(atoms: list[TechnicalAtom]) -> str:
    brain_exp = OperationalBrain()
    grouped = brain_exp.classify_by_taxonomy(atoms)
    lines = ["# Master Parameter Register\n"]
    for cat in TaxonomyCategory:
        cat_atoms = grouped.get(cat.value, [])
        lines.append(f"## {cat.value}\n")
        if not cat_atoms:
            lines.append("*No data extracted.*\n")
        else:
            for a in cat_atoms:
                unit = f" {a.unit}" if a.unit else ""
                lines.append(f"- **{a.parameter}**: {a.value}{unit}  *(source: {a.source_file})*")
            lines.append("")
    return "\n".join(lines)


def _build_pdf_bytes(md_text: str) -> bytes:
    """Try to build a PDF with fpdf2. Falls back to returning Markdown bytes."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        for line in md_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# "):
                pdf.set_font("Helvetica", "B", 16)
                pdf.set_fill_color(30, 30, 30)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(0, 12, stripped.lstrip("# ").strip(), new_x="LMARGIN", new_y="NEXT", fill=True)
                pdf.set_text_color(0, 0, 0)
            elif stripped.startswith("## "):
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_fill_color(50, 50, 50)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(0, 10, stripped.lstrip("# ").strip(), new_x="LMARGIN", new_y="NEXT", fill=True)
                pdf.set_text_color(0, 0, 0)
            elif stripped.startswith("- "):
                pdf.set_font("Helvetica", "", 10)
                # Remove bold markers for PDF
                clean = stripped.lstrip("- ").replace("**", "").replace("*", "")
                pdf.multi_cell(0, 6, f"  • {clean}")
            elif stripped:
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(0, 6, stripped.replace("**", "").replace("*", ""))
            else:
                pdf.ln(3)

        return bytes(pdf.output())
    except ImportError:
        return md_text.encode("utf-8")


with tab_export:
    st.subheader("Export")

    source_exp = st.session_state["resolved_atoms"] or st.session_state["atoms"]
    if not source_exp:
        st.info("No atoms yet. Run extraction first.")
    else:
        md_text = _build_markdown(source_exp)

        col_md, col_pdf = st.columns(2)
        with col_md:
            st.download_button(
                "Download Markdown",
                data=md_text,
                file_name="master_parameter_register.md",
                mime="text/markdown",
            )
        with col_pdf:
            pdf_bytes = _build_pdf_bytes(md_text)
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="master_parameter_register.pdf",
                mime="application/pdf",
            )

        with st.expander("Preview Markdown"):
            st.markdown(md_text)
