"""
app.py  –  The Operational Architect
Master Production Record Generator
"""

from __future__ import annotations

import html
import io
import logging
import os
import sys
import textwrap
from datetime import datetime, timezone

# Ensure the app directory is on sys.path so local modules are found
# (required for Streamlit Cloud where CWD may differ from the app directory)
_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

log = logging.getLogger(__name__)

import streamlit as st

from engine import (
    COST_TIERS,
    DEFAULT_CHUNK_CHARS,
    TAXONOMY_SECTIONS,
    ExtractionDiagnostics,
    FileExtractionDiag,
    OperationalBrain,
    TechnicalAtom,
    clear_extraction_cache,
    estimate_pipeline_cost,
    split_text,
    usage_stats,
)
from resolver import (
    Conflict,
    ResolutionState,
    apply_resolutions,
    deduplicate_atoms,
    detect_conflicts,
    flatten_taxonomy,
    infer_material_groups,
    resolve_conflict,
)
from self_healer import get_healer
from chemistry import (
    KitZone,
    SensoryType,
    ChemistryInstructionSet,
    ZONE_DESCRIPTIONS,
    ZONE_PHYSICAL,
    ZONE_ICONS,
    build_chemistry_instructions,
    is_chemistry_taxonomy,
)
from layman_v3 import (
    LaymanV3Brain,
    ValidationReport,
    fix_v3_document,
    validate_v3_document,
)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="The Operational Architect",
    page_icon="\U0001f3d7\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global ──────────────────────── */
    html, body, [data-testid="stApp"] {
        background-color: #0d0f14;
        color: #ffffff;
        font-family: 'IBM Plex Mono', 'Fira Code', monospace;
    }
    h1, h2, h3 { color: #ffffff; }
    p, span, li, label, div, td, th { color: #ffffff; }
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stText"],
    [data-testid="stCaptionContainer"],
    .stMarkdown, .stMarkdown p {
        color: #ffffff !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] div {
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
    }
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
    }

    /* ── Sidebar ─────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid #1e2330;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        color: #ffffff;
    }
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stSelectbox select {
        background-color: #1a1e2a;
        color: #ffffff;
        border: 1px solid #2d3346;
    }
    /* Streamlit BaseWeb selectbox overrides */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * {
        background-color: #1a1e2a !important;
        color: #ffffff !important;
        border-color: #2d3346 !important;
    }
    [data-testid="stSidebar"] .stSelectbox svg {
        fill: #ffffff !important;
    }
    /* Selectbox dropdown menu */
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [role="option"] {
        background-color: #1a1e2a !important;
        color: #ffffff !important;
    }
    [data-baseweb="popover"] [role="option"]:hover {
        background-color: #2d3346 !important;
    }
    /* File uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploader"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] *,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        background-color: #1a1e2a !important;
        color: #ffffff !important;
        border-color: #2d3346 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
        color: #9ca3af !important;
    }
    /* Text input in sidebar */
    [data-testid="stSidebar"] .stTextInput [data-baseweb="input"],
    [data-testid="stSidebar"] .stTextInput [data-baseweb="input"] input {
        background-color: #1a1e2a !important;
        color: #ffffff !important;
        border-color: #2d3346 !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* ── Cards ───────────────────────── */
    .mpr-card {
        background: #141720;
        border: 1px solid #1e2330;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .mpr-card-header {
        font-size: 0.7rem;
        color: #4a90d9;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .atom-row {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(30, 35, 48, 0.7);
        font-size: 0.82rem;
    }
    .atom-row:last-child { border-bottom: none; }
    .atom-row:hover { background: rgba(59, 130, 246, 0.04); border-radius: 4px; }
    .atom-param {
        color: #e2e8f0; flex: 0 0 40%; font-weight: 600;
        padding-left: 0.25rem;
    }
    .atom-value { color: #94a3b8; flex: 1; }
    .atom-tag   { flex: 0 0 70px; text-align: center; }
    .atom-src   { color: #4a5568; font-size: 0.68rem; white-space: nowrap;
                  flex: 0 0 100px; text-align: right; }
    .badge-prereq {
        background: #312e81; color: #a5b4fc;
        border-radius: 4px; padding: 1px 6px;
        font-size: 0.65rem; font-weight: 700;
        white-space: nowrap;
    }
    .badge-hoist {
        background: #1e3a2f; color: #6ee7b7;
        border-radius: 4px; padding: 1px 6px;
        font-size: 0.65rem; font-weight: 700;
        white-space: nowrap;
    }
    .material-group-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0 0.25rem 0;
        margin-top: 0.5rem;
        border-top: 1px solid #2d3346;
        color: #f59e0b;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .material-group-header:first-child {
        margin-top: 0;
        border-top: none;
    }
    .material-group-count {
        color: #6b7280;
        font-size: 0.68rem;
        font-weight: 400;
    }

    /* ── Gap audit ───────────────────── */
    .gap-badge {
        display: inline-block;
        background: #7f1d1d; color: #fca5a5;
        border-radius: 5px; padding: 3px 10px;
        margin: 3px; font-size: 0.75rem; font-weight: 700;
    }
    .gap-ok-badge {
        display: inline-block;
        background: #14532d; color: #86efac;
        border-radius: 5px; padding: 3px 10px;
        margin: 3px; font-size: 0.75rem;
    }

    /* ── Conflict panel ──────────────── */
    .conflict-header {
        color: #f87171; font-weight: 700;
        font-size: 0.8rem; letter-spacing: 0.08em;
    }

    /* ── Buttons ─────────────────────── */
    .stButton > button {
        background-color: #1e40af;
        color: white;
        border: none;
        border-radius: 6px;
        font-family: inherit;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #2563eb; }

    /* ── Progress bar ────────────────── */
    .stProgress > div > div { background-color: #3b82f6; }

    /* ── Chemistry Zones ────────────── */
    .chem-zone-card {
        background: #141720;
        border: 1px solid #1e2330;
        border-radius: 12px;
        padding: 0;
        margin-bottom: 1.25rem;
        overflow: hidden;
    }
    .chem-zone-a { border-left: 4px solid #3b82f6; }
    .chem-zone-b { border-left: 4px solid #8b5cf6; }
    .chem-zone-c { border-left: 4px solid #f59e0b; }
    .chem-zone-d { border-left: 4px solid #6b7280; }

    /* Zone header with icon */
    .chem-zone-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.25rem 0.75rem;
        background: linear-gradient(135deg, #141720 0%, #1a1e2e 100%);
    }
    .chem-zone-icon {
        flex-shrink: 0;
        width: 64px;
        height: 64px;
    }
    .chem-zone-header-text { flex: 1; }
    .chem-zone-label {
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .chem-zone-a .chem-zone-label { color: #3b82f6; }
    .chem-zone-b .chem-zone-label { color: #8b5cf6; }
    .chem-zone-c .chem-zone-label { color: #f59e0b; }
    .chem-zone-d .chem-zone-label { color: #6b7280; }
    .chem-zone-desc {
        color: #9ca3af;
        font-size: 0.78rem;
        margin-bottom: 0;
        font-style: italic;
        line-height: 1.4;
    }
    .chem-zone-physical {
        color: #cbd5e1;
        font-size: 0.78rem;
        margin-top: 0.25rem;
        line-height: 1.4;
    }

    /* Zone items table */
    .chem-zone-items {
        padding: 0 1.25rem 1rem;
    }
    .chem-zone-items-header {
        display: flex;
        gap: 0.5rem;
        padding: 0.5rem 0;
        border-bottom: 2px solid #1e2740;
        margin-bottom: 0.25rem;
    }
    .chem-zone-items-header span {
        font-size: 0.65rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #4a5568;
        font-weight: 600;
    }
    .chem-col-item { flex: 0 0 40%; }
    .chem-col-spec { flex: 1; }
    .chem-col-tag  { flex: 0 0 70px; text-align: center; }
    .chem-col-src  { flex: 0 0 100px; text-align: right; }

    /* ── Action-Result Table ─────────── */
    .ar-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.75rem 0;
    }
    .ar-table th {
        background: #1a1e2a;
        color: #4a90d9;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.5rem 0.75rem;
        text-align: left;
        border-bottom: 2px solid #2d3346;
    }
    .ar-table td {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #1e2330;
        font-size: 0.82rem;
        vertical-align: top;
    }
    .ar-step-num {
        color: #4a90d9;
        font-weight: 700;
        white-space: nowrap;
        width: 40px;
    }
    .ar-action { color: #e2e8f0; }
    .ar-result { color: #6ee7b7; font-style: italic; }

    /* ── Red Box (hazardous steps) ──── */
    .red-box {
        border: 2px solid #dc2626;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        background: rgba(220, 38, 38, 0.08);
    }
    .red-box-label {
        color: #fca5a5;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .red-box td { border-bottom-color: #7f1d1d !important; }

    /* ── Sensory Badges ─────────────── */
    .sensory-badge {
        display: inline-block;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.65rem;
        font-weight: 700;
        margin: 2px 3px;
        white-space: nowrap;
    }
    .sensory-visual   { background: #1e3a5f; color: #93c5fd; }
    .sensory-auditory  { background: #3b1f5e; color: #c4b5fd; }
    .sensory-tactile   { background: #3b2f1e; color: #fcd34d; }
    .sensory-olfactory { background: #1e3a2f; color: #86efac; }

    /* ── Safe Pause Point ───────────── */
    .safe-pause {
        background: #14532d;
        border: 1px dashed #22c55e;
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        color: #86efac;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.4rem 0;
        text-align: center;
    }

    /* ── Troubleshooting Sidebar ────── */
    .ts-card {
        background: #1a1520;
        border: 1px solid #2d2340;
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        margin: 0.4rem 0;
        font-size: 0.78rem;
    }
    .ts-card-critical {
        border-color: #7f1d1d;
        background: #1a1215;
    }
    .ts-label {
        font-size: 0.65rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .ts-label-warning { color: #f59e0b; }
    .ts-label-critical { color: #f87171; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

def _fresh_defaults() -> dict:
    return {
        "raw_atoms": [],            # List[TechnicalAtom]  – all extracted
        "taxonomy": {},             # Dict[str, List[TechnicalAtom]]
        "resolution_state": None,   # ResolutionState
        "resolved_taxonomy": {},    # Dict after conflict resolution
        "extraction_done": False,
        "resolution_done": False,
        "file_texts": {},           # filename → extracted text
        "extraction_diagnostics": None,  # ExtractionDiagnostics
    }

for k, v in _fresh_defaults().items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
# Sidebar – Configuration
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## \U0001f3d7\ufe0f Operational Architect")
    st.markdown("---")

    provider = st.selectbox("LLM Provider", ["Anthropic", "OpenAI"])

    # Pre-populate from Streamlit secrets if available
    _secret_key = ""
    if provider == "Anthropic":
        _secret_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    elif provider == "OpenAI":
        _secret_key = st.secrets.get("OPENAI_API_KEY", "")

    api_key = st.text_input(
        "API Key",
        value=_secret_key,
        type="password",
        placeholder="sk-ant-... or sk-...",
    )

    # ── Early API-key validation ──
    _key_ok = True
    if api_key:
        if provider == "Anthropic" and not api_key.startswith("sk-ant-"):
            st.warning("Anthropic keys typically start with `sk-ant-`. Double-check your key.")
            _key_ok = False
        elif provider == "OpenAI" and not api_key.startswith("sk-"):
            st.warning("OpenAI keys typically start with `sk-`. Double-check your key.")
            _key_ok = False
        elif provider == "OpenAI" and api_key.startswith("sk-ant-"):
            st.warning("This looks like an Anthropic key, but OpenAI is selected.")
            _key_ok = False

    model_override = st.text_input(
        "Model (optional)",
        placeholder="claude-sonnet-4-20250514",
        help="Leave blank to use the recommended default for each provider.",
    )

    cost_tier = st.selectbox(
        "Cost / Speed Tier",
        ["quality", "balanced", "economy"],
        format_func=lambda t: {
            "quality": "Quality  —  best extraction accuracy",
            "balanced": "Balanced  —  good quality, much cheaper",
            "economy": "Economy  —  fastest & cheapest",
        }[t],
        help="Controls which model is used when 'Model' is left blank.",
    )

    # Show the effective model for transparency
    _provider_key = provider.strip().lower()
    _tier_info = COST_TIERS.get(_provider_key, {}).get(cost_tier, {})
    if not model_override and _tier_info:
        st.caption(f"Using: **{_tier_info.get('label', '')}**")

    st.markdown("---")
    st.markdown("### \U0001f4c2 Upload Source Files")
    uploaded_files = st.file_uploader(
        "TXT / MD / PDF files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#4a5568;'>"
        "Each file is independently processed then merged. "
        "Conflicts must be manually resolved before export."
        "</small>",
        unsafe_allow_html=True,
    )

    if st.button("\U0001f5d1\ufe0f Reset Session", use_container_width=True):
        clear_extraction_cache()
        usage_stats.reset()
        for k, v in _fresh_defaults().items():
            st.session_state[k] = v
        cleared = clear_extraction_cache()
        usage_stats.reset()
        if cleared:
            log.info("Reset session: cleared %d cached extraction(s)", cleared)
        st.rerun()

    # ── Self-Healing Dashboard ─────────────
    st.markdown("---")
    st.markdown("### Self-Healing")
    healer_sidebar = get_healer()
    stats = healer_sidebar.get_stats()

    if stats["total_runs"] > 0:
        st.markdown(
            f"<small>"
            f"Runs: <b>{stats['total_runs']}</b> &nbsp;|&nbsp; "
            f"Success: <b>{stats['success_rate']}</b> &nbsp;|&nbsp; "
            f"Recoveries: <b>{stats['total_recoveries']}</b>"
            f"</small>",
            unsafe_allow_html=True,
        )
        if stats["top_errors"]:
            st.markdown("<small><b>Top errors:</b></small>", unsafe_allow_html=True)
            for err_type, count in stats["top_errors"][:3]:
                st.markdown(
                    f"<small style='color:#f87171;'>&nbsp;&nbsp;{err_type}: {count}</small>",
                    unsafe_allow_html=True,
                )
        if stats["top_strategies"]:
            st.markdown("<small><b>Learned strategies:</b></small>", unsafe_allow_html=True)
            for s in stats["top_strategies"][:3]:
                st.markdown(
                    f"<small style='color:#6ee7b7;'>"
                    f"&nbsp;&nbsp;{s['name']}: {s['success_rate']} ({s['uses']} uses)"
                    f"</small>",
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            "<small style='color:#4a5568;'>No run history yet. "
            "Errors and recoveries will appear here.</small>",
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def read_file(uploaded) -> str:
    """Extract text from TXT, MD, or PDF upload.

    Self-healing: records file-read outcomes and applies learned encoding
    strategies on failure.
    """
    healer = get_healer()
    name = uploaded.name.lower()

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(uploaded.read()))
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            healer.record_success("file_read", context={"filename": uploaded.name, "type": "pdf"})
            return text
        except Exception as e:
            healer.record_failure("file_read", e, context={"filename": uploaded.name, "type": "pdf"})
            # Self-heal: try reading raw bytes as fallback
            try:
                uploaded.seek(0)
                raw = uploaded.read()
                text = raw.decode("latin-1", errors="replace")
                if len(text.strip()) > 50:
                    healer.record_success(
                        "file_read",
                        context={"filename": uploaded.name, "type": "pdf_fallback"},
                        recovery_strategy="try_alternate_encoding",
                    )
                    healer.learn_strategy("file_read", e, "try_alternate_encoding", succeeded=True)
                    return text
            except Exception as fallback_err:
                log.warning(
                    "PDF fallback read also failed for %s: %s",
                    uploaded.name,
                    fallback_err,
                )
                healer.record_failure(
                    "file_read",
                    fallback_err,
                    context={"filename": uploaded.name, "type": "pdf_fallback"},
                )
            return f"[PDF read error: {e}]"
    else:
        raw = uploaded.read()
        # Check history for a preferred encoding that worked before
        preferred_encodings = ["utf-8", "latin-1", "cp1252"]
        for record in reversed(healer.history[-50:]):
            if (
                record.operation == "file_read"
                and record.success
                and record.context.get("encoding")
                and record.context["encoding"] not in preferred_encodings[:1]
            ):
                # Move the historically successful encoding to front
                enc = record.context["encoding"]
                if enc in preferred_encodings:
                    preferred_encodings.remove(enc)
                    preferred_encodings.insert(0, enc)
                break

        for enc in preferred_encodings:
            try:
                text = raw.decode(enc)
                healer.record_success(
                    "file_read",
                    context={"filename": uploaded.name, "type": "text", "encoding": enc},
                )
                return text
            except UnicodeDecodeError as e:
                healer.record_failure(
                    "file_read", e,
                    context={"filename": uploaded.name, "type": "text", "encoding": enc},
                )
                continue
        return raw.decode("utf-8", errors="replace")


def _atom_html(atom: TechnicalAtom) -> str:
    badges = ""
    if atom.is_prerequisite:
        badges += '<span class="badge-prereq">PREREQ</span> '
    if atom.hoisted:
        badges += '<span class="badge-hoist">HOISTED</span> '
    param = html.escape(atom.parameter)
    value = html.escape(atom.value)
    src = html.escape(atom.source_file)
    return (
        f'<div class="atom-row">'
        f'  <span class="atom-param">{param}</span>'
        f'  <span class="atom-value">{value}</span>'
        f'  <span class="atom-tag">{badges if badges else ""}</span>'
        f'  <span class="atom-src">{src}</span>'
        f'</div>'
    )


def _group_atoms_by_material(atoms: list[TechnicalAtom]) -> list[tuple[str | None, list[TechnicalAtom]]]:
    """Group atoms by material_group, preserving order.

    Returns a list of (group_name, atoms) tuples. Ungrouped atoms have
    group_name=None and appear first.
    """
    from collections import OrderedDict

    ungrouped: list[TechnicalAtom] = []
    groups: OrderedDict[str, list[TechnicalAtom]] = OrderedDict()

    for atom in atoms:
        if atom.material_group:
            groups.setdefault(atom.material_group, []).append(atom)
        else:
            ungrouped.append(atom)

    result: list[tuple[str | None, list[TechnicalAtom]]] = []
    if ungrouped:
        result.append((None, ungrouped))
    for name, group_atoms in groups.items():
        result.append((name, group_atoms))
    return result


def render_taxonomy(taxonomy: dict[str, list[TechnicalAtom]]) -> None:
    for section in TAXONOMY_SECTIONS:
        atoms = taxonomy.get(section, [])
        count_label = f"({len(atoms)} atoms)" if atoms else "\u26a0\ufe0f EMPTY"
        color = "#4a90d9" if atoms else "#dc2626"
        with st.expander(f"{section}  \u2014  {count_label}", expanded=bool(atoms)):
            if not atoms:
                st.markdown(
                    "<small style='color:#dc2626;'>No data extracted for this section.</small>",
                    unsafe_allow_html=True,
                )
            else:
                groups = _group_atoms_by_material(atoms)
                rendered = '<div class="mpr-card">'
                for group_name, group_atoms in groups:
                    if group_name:
                        safe_name = html.escape(group_name)
                        rendered += (
                            f'<div class="material-group-header">'
                            f'{safe_name}'
                            f'<span class="material-group-count">'
                            f'{len(group_atoms)} properties</span></div>'
                        )
                    for atom in group_atoms:
                        rendered += _atom_html(atom)
                rendered += "</div>"
                st.markdown(rendered, unsafe_allow_html=True)


def render_gap_audit(taxonomy: dict[str, list[TechnicalAtom]]) -> None:
    gaps = [
        s
        for s in TAXONOMY_SECTIONS
        if s != "19. Gap Audit" and not taxonomy.get(s)
    ]
    present = [s for s in TAXONOMY_SECTIONS if s != "19. Gap Audit" and taxonomy.get(s)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sections", len(TAXONOMY_SECTIONS) - 1)
    col2.metric("Populated", len(present), delta=f"+{len(present)}", delta_color="normal")
    col3.metric("Missing \u26a0\ufe0f", len(gaps), delta=f"-{len(gaps)}", delta_color="inverse")

    if gaps:
        st.markdown("**Missing sections:**")
        html = "".join(f'<span class="gap-badge">{g}</span>' for g in gaps)
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.success("\u2705 All 19 operational sections populated.")

    if present:
        st.markdown("**Populated sections:**")
        html = "".join(f'<span class="gap-ok-badge">{p}</span>' for p in present)
        st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Export helpers
# ──────────────────────────────────────────────

def build_markdown(taxonomy: dict[str, list[TechnicalAtom]], title: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {title}",
        f"> Generated by The Operational Architect \u00b7 {ts}",
        "",
        "---",
        "",
    ]
    for section in TAXONOMY_SECTIONS:
        atoms = taxonomy.get(section, [])
        lines.append(f"## {section}")
        if not atoms:
            lines.append("*No data extracted for this section.*")
        else:
            groups = _group_atoms_by_material(atoms)
            for group_name, group_atoms in groups:
                if group_name:
                    lines.append(f"\n### {group_name}")
                for atom in group_atoms:
                    prereq = " `[PREREQ]`" if atom.is_prerequisite else ""
                    hoisted = " `[HOISTED]`" if atom.hoisted else ""
                    lines.append(
                        f"- **{atom.parameter}**: {atom.value}{prereq}{hoisted}  "
                        f"*(source: {atom.source_file})*"
                    )
        lines.append("")
    return "\n".join(lines)


def build_pdf(taxonomy: dict[str, list[TechnicalAtom]], title: str) -> bytes:
    try:
        from fpdf import FPDF

        class MPR_PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 9)
                self.set_text_color(74, 144, 217)
                self.cell(0, 6, "THE OPERATIONAL ARCHITECT  \u00b7  MASTER PRODUCTION RECORD", align="C")
                self.ln(3)
                self.set_draw_color(30, 35, 48)
                self.line(10, self.get_y(), 200, self.get_y())
                self.ln(3)

            def footer(self):
                self.set_y(-12)
                self.set_font("Helvetica", "", 7)
                self.set_text_color(100, 110, 130)
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                self.cell(0, 6, f"Generated {ts}  |  Page {self.page_no()}", align="C")

        pdf = MPR_PDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_margins(14, 14, 14)
        pdf.add_page()

        # Title
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(226, 232, 240)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 110, 130)
        pdf.cell(0, 6, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        for section in TAXONOMY_SECTIONS:
            atoms = taxonomy.get(section, [])

            # Section header
            pdf.set_fill_color(20, 23, 32)
            pdf.set_draw_color(30, 35, 48)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(74, 144, 217)
            pdf.cell(0, 7, section, fill=True, border=1, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

            if not atoms:
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(220, 38, 38)
                pdf.cell(0, 5, "  [NO DATA \u2013 GAP IDENTIFIED]", new_x="LMARGIN", new_y="NEXT")
            else:
                groups = _group_atoms_by_material(atoms)
                for group_name, group_atoms in groups:
                    if group_name:
                        pdf.set_font("Helvetica", "B", 9)
                        pdf.set_text_color(245, 158, 11)
                        pdf.cell(0, 6, f"  {group_name}", new_x="LMARGIN", new_y="NEXT")
                    for atom in group_atoms:
                        badges = ""
                        if atom.is_prerequisite:
                            badges += " [PREREQ]"
                        if atom.hoisted:
                            badges += " [HOISTED]"
                        src = f"  (src: {atom.source_file})"

                        # Parameter
                        pdf.set_font("Helvetica", "B", 8)
                        pdf.set_text_color(125, 211, 252)
                        param_text = f"  {atom.parameter}"
                        pdf.multi_cell(55, 4.5, param_text, new_x="RIGHT", new_y="LAST")

                        # Value
                        pdf.set_font("Helvetica", "", 8)
                        pdf.set_text_color(201, 209, 224)
                        value_text = atom.value + badges
                        # Wrap long values
                        x_after_param = pdf.get_x()
                        remaining_w = pdf.w - pdf.r_margin - x_after_param
                        pdf.multi_cell(remaining_w - 28, 4.5, value_text, new_x="RIGHT", new_y="LAST")

                        # Source
                        pdf.set_font("Helvetica", "I", 7)
                        pdf.set_text_color(74, 85, 104)
                        pdf.multi_cell(28, 4.5, src, new_x="LMARGIN", new_y="NEXT")
                        pdf.set_draw_color(26, 30, 42)
                        pdf.line(14, pdf.get_y(), pdf.w - 14, pdf.get_y())

            pdf.ln(3)

        return bytes(pdf.output())

    except ImportError as e:
        st.error(
            "PDF export requires the `fpdf2` package. "
            "Install it with: `pip install fpdf2`"
        )
        md = build_markdown(taxonomy, title)
        return md.encode("utf-8")
    except Exception as e:
        # Self-healing: record PDF failure, then apply markdown fallback
        healer = get_healer()
        healer.record_failure("pdf_export", e, context={"title": title})
        st.warning(f"PDF generation failed ({e}). Falling back to Markdown bytes.")
        md = build_markdown(taxonomy, title)
        fallback_bytes = md.encode("utf-8")
        # Record the fallback as a successful recovery
        healer.record_success(
            "pdf_export",
            context={"title": title, "fallback": "markdown"},
            recovery_strategy="fallback_pdf_to_markdown",
        )
        healer.learn_strategy("pdf_export", e, "fallback_pdf_to_markdown", succeeded=True)
        return fallback_bytes


# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────

st.markdown("# \U0001f3d7\ufe0f The Operational Architect")
st.markdown(
    "<p style='color:#4a5568; font-size:0.85rem;'>"
    "Schema-First \u00b7 Zero-Loss \u00b7 Phase-Gated Master Production Record Generator"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Tab Layout ──────────────────────────────────

tab_extract, tab_resolve, tab_mpr, tab_chem, tab_v3, tab_gap, tab_export = st.tabs(
    ["\u2460 Extract", "\u2461 Resolve Conflicts", "\u2462 MPR View",
     "\u2463 Chemistry Instructions", "\u2464 Layman Doc v3",
     "\u2465 Gap Audit", "\u2466 Export"]
)

# ════════════════════════════════════════════════
# TAB 1 – EXTRACTION
# ════════════════════════════════════════════════

with tab_extract:
    st.markdown("### Step 1 \u00b7 Upload & Extract")

    if not uploaded_files:
        st.info("\U0001f448 Upload one or more TXT / MD / PDF files in the sidebar to begin.")
    else:
        st.markdown(f"**{len(uploaded_files)} file(s) ready:**")
        for f in uploaded_files:
            size_note = (
                " \u2014 large file, will be split into chunks"
                if f.size > DEFAULT_CHUNK_CHARS
                else ""
            )
            st.markdown(f"- `{f.name}` ({f.size:,} bytes){size_note}")

        # ── Pre-run cost / time estimate ──
        _forensic_texts = {}
        for f in uploaded_files:
            f.seek(0)
            try:
                _forensic_texts[f.name] = f.read().decode("utf-8", errors="replace")
            except Exception:
                _forensic_texts[f.name] = ""
            f.seek(0)
        if _forensic_texts:
            f_est = estimate_pipeline_cost(
                _forensic_texts,
                provider=provider.lower(),
                tier=cost_tier,
                pipeline="forensic",
                model_override=model_override or None,
            )
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Est. API Calls", f_est["api_calls"])
            fc2.metric("Est. Time", f_est["time_display"])
            fc3.metric("Est. Cost", f"${f_est['cost_usd']:.3f}")

        run_extraction = st.button("\U0001f52c Run Forensic Extraction", use_container_width=True)

        if run_extraction:
            if not api_key:
                st.error("\u26d4 No API key configured. Add your key in the sidebar.")
            elif not _key_ok:
                st.error("\u26d4 API key looks invalid for the selected provider. Fix it in the sidebar.")
            else:
                st.session_state["extraction_done"] = False
                st.session_state["resolution_done"] = False
                st.session_state["extraction_diagnostics"] = None
                all_atoms: list[TechnicalAtom] = []
                file_texts: dict[str, str] = {}
                diagnostics = ExtractionDiagnostics()

                brain = OperationalBrain(
                    provider=provider,
                    api_key=api_key,
                    model=model_override or None,
                    tier=cost_tier,
                )

                n_files = len(uploaded_files)
                progress = st.progress(0, text="Starting extraction\u2026")
                status_area = st.empty()
                failed_files: list[str] = []

                healer = get_healer()

                for i, uf in enumerate(uploaded_files):
                    # Progress: show file N/M and a sub-step
                    base_pct = int((i / n_files) * 100)
                    progress.progress(base_pct, text=f"[{i+1}/{n_files}] Reading `{uf.name}`\u2026")
                    status_area.markdown(
                        f"<small style='color:#4a90d9;'>Reading {uf.name}\u2026</small>",
                        unsafe_allow_html=True,
                    )

                    file_diag = FileExtractionDiag(filename=uf.name)
                    max_file_retries = 2
                    succeeded = False
                    last_extraction_error = None

                    for attempt in range(max_file_retries + 1):
                        try:
                            uf.seek(0)  # Reset file pointer for retries
                            text = read_file(uf)
                            if text.startswith("[PDF read error"):
                                file_diag.file_read_ok = False
                                file_diag.file_read_error = text
                                st.warning(f"Could not read `{uf.name}`. It may be a scanned/image-only PDF.")
                                failed_files.append(uf.name)
                                break
                            if len(text.strip()) < 20:
                                file_diag.file_read_ok = True
                                file_diag.text_length = len(text)
                                file_diag.warnings.append(
                                    f"File too short ({len(text.strip())} chars after stripping whitespace)"
                                )
                                st.warning(f"`{uf.name}` appears to be empty or too short to extract from.")
                                failed_files.append(uf.name)
                                break
                            file_texts[uf.name] = text
                            file_diag.text_length = len(text)

                            # Sub-step: extraction via LLM
                            mid_pct = int(((i + 0.5) / n_files) * 100)
                            n_chunks = len(split_text(text))
                            chunk_note = (
                                f" (split into {n_chunks} chunks)"
                                if n_chunks > 1
                                else ""
                            )
                            progress.progress(mid_pct, text=f"[{i+1}/{n_files}] Extracting specs from `{uf.name}`{chunk_note}\u2026")
                            status_area.markdown(
                                f"<small style='color:#4a90d9;'>Calling LLM for {uf.name}{chunk_note}\u2026 (this may take a moment)</small>",
                                unsafe_allow_html=True,
                            )

                            atoms = brain.extract_atoms(text, uf.name, diag=file_diag)
                            all_atoms.extend(atoms)

                            recovery_note = ""
                            if attempt > 0 and last_extraction_error:
                                recovery_note = f" (self-healed after {attempt} retry)"
                                healer.record_success(
                                    "file_extraction",
                                    context={"filename": uf.name, "attempt": attempt},
                                    recovery_strategy="retry_with_backoff",
                                )
                                healer.learn_strategy(
                                    "file_extraction", last_extraction_error,
                                    "retry_with_backoff", succeeded=True,
                                )

                            if len(atoms) == 0:
                                status_area.markdown(
                                    f"<small style='color:#f59e0b;'>\u26a0 {uf.name} \u2192 0 atoms extracted. "
                                    f"The document may not contain recognisable technical specifications.</small>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                status_area.markdown(
                                    f"<small style='color:#6ee7b7;'>"
                                    f"\u2713 {uf.name} \u2192 {len(atoms)} atoms{recovery_note}"
                                    f"</small>",
                                    unsafe_allow_html=True,
                                )
                            succeeded = True
                            break

                        except Exception as e:
                            last_extraction_error = e
                            file_diag.llm_call_ok = False
                            file_diag.llm_call_error = f"{type(e).__name__}: {e}"
                            healer.record_failure(
                                "file_extraction", e,
                                context={"filename": uf.name, "attempt": attempt},
                            )
                            if attempt < max_file_retries:
                                strategies = healer.get_recovery_strategies("file_extraction", e)
                                wait = 2 ** (attempt + 1)
                                status_area.markdown(
                                    f"<small style='color:#f59e0b;'>"
                                    f"\u26a0 {uf.name} failed (attempt {attempt+1}), "
                                    f"retrying in {wait}s "
                                    f"[strategy: {strategies[0] if strategies else 'backoff'}]"
                                    f"</small>",
                                    unsafe_allow_html=True,
                                )
                                import time as _time
                                _time.sleep(wait)

                    diagnostics.add(file_diag)

                    if not succeeded and uf.name not in failed_files:
                        failed_files.append(uf.name)
                        err_msg = str(healer.get_recent_failures(1)[0].error_message) if healer.get_recent_failures(1) else "Unknown error"
                        # Translate common API errors into user-friendly messages
                        if "authentication" in err_msg.lower() or "401" in err_msg:
                            st.error(f"`{uf.name}`: API key is invalid or expired. Check your key in the sidebar.")
                        elif "rate" in err_msg.lower() or "429" in err_msg:
                            st.error(f"`{uf.name}`: API rate limit hit after {max_file_retries+1} attempts.")
                        elif "overloaded" in err_msg.lower() or "503" in err_msg:
                            st.error(f"`{uf.name}`: The AI service is temporarily overloaded after {max_file_retries+1} attempts.")
                        elif "timeout" in err_msg.lower():
                            st.error(f"`{uf.name}`: Request timed out. The document may be too large.")
                        elif "400" in err_msg or "bad request" in err_msg.lower():
                            st.error(
                                f"`{uf.name}`: Bad request (HTTP 400). This usually means the document "
                                f"is too large, the prompt exceeds the model's context window, or the "
                                f"API key doesn't have access to the selected model. "
                                f"Try a smaller file or switch to a different model/tier."
                            )
                        else:
                            st.error(f"Error processing `{uf.name}` after {max_file_retries+1} attempts. Check the Self-Healing Dashboard.")

                progress.progress(100, text="Extraction complete.")

                # Summary diagnostics
                if failed_files:
                    st.warning(
                        f"{len(failed_files)} file(s) could not be processed: "
                        + ", ".join(f"`{f}`" for f in failed_files)
                        + ". Successfully extracted from the remaining files."
                    )
                if len(all_atoms) == 0 and not failed_files:
                    st.warning(
                        "No technical specifications were extracted from any file. "
                        "This can happen if the documents don't contain structured technical data, "
                        "or if the LLM couldn't parse them. Try uploading clearer source documents."
                    )

                # Show cost / usage summary
                stats = usage_stats.summary()
                if stats["calls"] > 0 or stats["cache_hits"] > 0:
                    cost_col1, cost_col2, cost_col3 = st.columns(3)
                    cost_col1.metric("API Calls", stats["calls"])
                    cost_col2.metric("Cache Hits", stats["cache_hits"])
                    cost_col3.metric("Est. Cost", f"${stats['estimated_cost_usd']:.4f}")
                    if stats["input_tokens"] > 0:
                        st.caption(
                            f"Tokens: {stats['input_tokens']:,} in / {stats['output_tokens']:,} out"
                        )

                # Deduplicate exact-match atoms across files and infer groups
                pre_dedup_count = len(all_atoms)
                all_atoms = deduplicate_atoms(all_atoms)
                all_atoms = infer_material_groups(all_atoms)
                dedup_removed = pre_dedup_count - len(all_atoms)

                taxonomy = brain.organize_by_taxonomy(all_atoms)
                resolution_state = detect_conflicts(all_atoms)

                st.session_state["dedup_removed"] = dedup_removed
                st.session_state["raw_atoms"] = all_atoms
                st.session_state["taxonomy"] = taxonomy
                st.session_state["file_texts"] = file_texts
                st.session_state["resolution_state"] = resolution_state
                st.session_state["extraction_done"] = True
                st.session_state["extraction_diagnostics"] = diagnostics

                # If no conflicts, auto-resolve
                if resolution_state.total_count == 0:
                    st.session_state["resolved_taxonomy"] = taxonomy
                    st.session_state["resolution_done"] = True

                st.rerun()

        if st.session_state["extraction_done"]:
            total = len(st.session_state["raw_atoms"])
            conflicts = (
                st.session_state["resolution_state"].total_count
                if st.session_state["resolution_state"]
                else 0
            )
            dedup_removed = st.session_state.get("dedup_removed", 0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Atoms", total)
            c2.metric("Files Processed", len(st.session_state["file_texts"]))
            c3.metric("Duplicates Merged", dedup_removed)
            c4.metric("Conflicts Detected", conflicts, delta=None)

            if conflicts > 0:
                st.warning(
                    f"\u26a0\ufe0f {conflicts} conflict(s) detected. "
                    "Switch to the **\u2461 Resolve Conflicts** tab to proceed."
                )
            else:
                st.success("\u2705 No conflicts. Navigate to **\u2462 MPR View** or **\u2465 Export**.")

            # ── Extraction Diagnostics Panel ──────────────
            diag_data: ExtractionDiagnostics | None = st.session_state.get(
                "extraction_diagnostics"
            )
            if diag_data is not None:
                show_diag = diag_data.has_any_problems() or total == 0
                label = (
                    "Extraction Diagnostics (issues detected)"
                    if show_diag
                    else "Extraction Diagnostics"
                )
                with st.expander(label, expanded=show_diag):
                    if total == 0:
                        st.error(
                            "No atoms were extracted. The diagnostics below "
                            "show what happened at each stage of the pipeline."
                        )
                    for fd in diag_data.file_diags:
                        icon = "x" if fd.has_problems() else "white_check_mark"
                        with st.expander(
                            f":{icon}: `{fd.filename}`",
                            expanded=fd.has_problems(),
                        ):
                            for line in fd.summary_lines():
                                st.text(line)

# ════════════════════════════════════════════════
# TAB 2 – CONFLICT RESOLUTION
# ════════════════════════════════════════════════

with tab_resolve:
    st.markdown("### Step 2 \u00b7 Resolve Data Conflicts")

    rs: ResolutionState | None = st.session_state.get("resolution_state")

    if not st.session_state["extraction_done"]:
        st.info("Run extraction first (Tab \u2460).")

    elif rs is None or rs.total_count == 0:
        st.success("\u2705 No conflicts detected. All source files are in agreement.")

    else:
        unresolved = rs.unresolved_count
        st.markdown(
            f"<p class='conflict-header'>"
            f"\u26a0\ufe0f  {unresolved} of {rs.total_count} conflict(s) awaiting resolution"
            f"</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "For each conflict, select which source file is the **Source of Truth**. "
            "The losing value will be discarded."
        )
        st.markdown("---")

        for conflict in rs.conflicts:
            icon = "\u2705" if conflict.resolved else "\U0001f534"
            with st.expander(
                f"{icon} {conflict.conflict_id}  \u00b7  {conflict.short_label()}",
                expanded=not conflict.resolved,
            ):
                st.markdown(
                    f"<div class='mpr-card-header'>"
                    f"CATEGORY: {conflict.category}  &nbsp;|&nbsp;  "
                    f"PARAMETER: {conflict.parameter}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                options = []
                for idx, (src, val, atom) in enumerate(conflict.candidates):
                    ellipsis = "\u2026" if len(val) > 120 else ""
                    label = f"[{src}]  \u2192  {val[:120]}{ellipsis}"
                    options.append(label)

                if conflict.resolved and conflict.resolved_atom:
                    st.markdown(
                        f"**Resolved \u2192 Source of Truth:** `{conflict.resolved_atom.source_file}`  "
                        f"\u2192 _{conflict.resolved_atom.value}_"
                    )
                    if st.button(f"\u21a9 Undo {conflict.conflict_id}", key=f"undo_{conflict.conflict_id}"):
                        conflict.resolved = False
                        conflict.resolved_atom = None
                        st.session_state["resolution_done"] = False
                        st.rerun()
                else:
                    choice = st.radio(
                        "Select Source of Truth:",
                        options=options,
                        index=None,
                        key=f"radio_{conflict.conflict_id}",
                    )
                    if choice is not None:
                        chosen_idx = options.index(choice)
                        resolve_conflict(conflict, chosen_idx)
                        st.session_state["resolution_done"] = False
                        st.rerun()

        st.markdown("---")
        all_done = rs.all_resolved
        if all_done:
            # Show a summary of chosen resolutions before finalizing
            st.markdown("**Resolution summary** — review before finalizing:")
            for c in rs.conflicts:
                if c.resolved_atom:
                    st.markdown(
                        f"- **{c.parameter}** ({c.category}): "
                        f"using value from `{c.resolved_atom.source_file}` "
                        f"— _{c.resolved_atom.value[:80]}{'…' if len(c.resolved_atom.value) > 80 else ''}_"
                    )
            st.markdown("")
            if st.button("\u2705 Finalize Resolutions & Build MPR", use_container_width=True):
                resolved = apply_resolutions(st.session_state["taxonomy"], rs)
                st.session_state["resolved_taxonomy"] = resolved
                st.session_state["resolution_done"] = True
                st.success("Resolutions applied. Navigate to **\u2462 MPR View** or **\u2465 Export**.")
                st.rerun()
        else:
            st.warning(
                f"{rs.unresolved_count} conflict(s) still unresolved. "
                "Resolve all before finalizing."
            )

# ════════════════════════════════════════════════
# TAB 3 – MPR VIEW
# ════════════════════════════════════════════════

with tab_mpr:
    st.markdown("### Step 3 \u00b7 Master Production Record")

    taxonomy_to_show = st.session_state.get("resolved_taxonomy") or st.session_state.get("taxonomy")

    if not taxonomy_to_show:
        st.info("Run extraction first (Tab \u2460).")
    else:
        rs = st.session_state.get("resolution_state")
        if rs and rs.total_count > 0 and not st.session_state.get("resolution_done"):
            st.warning(
                "\u26a0\ufe0f This MPR is **pre-resolution**. "
                f"{rs.unresolved_count} conflict(s) remain. "
                "Resolve them in Tab \u2461 for the final document."
            )

        all_atoms_flat = flatten_taxonomy(taxonomy_to_show)
        prereqs = [a for a in all_atoms_flat if a.is_prerequisite]
        hoisted = [a for a in all_atoms_flat if a.hoisted]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Atoms", len(all_atoms_flat))
        c2.metric("Prerequisites", len(prereqs))
        c3.metric("Hoisted Items", len(hoisted))

        st.markdown("---")
        render_taxonomy(taxonomy_to_show)

# ════════════════════════════════════════════════
# TAB 4 – CHEMISTRY INSTRUCTIONS
# ════════════════════════════════════════════════

with tab_chem:
    st.markdown("### Chemistry Instructions Designer")
    st.markdown(
        "<p style='color:#4a5568; font-size:0.82rem;'>"
        "Kit-Based \u00b7 Sensory-Anchored \u00b7 Safety-Embedded procedure view"
        "</p>",
        unsafe_allow_html=True,
    )

    chem_taxonomy = (
        st.session_state.get("resolved_taxonomy")
        or st.session_state.get("taxonomy")
    )

    if not chem_taxonomy:
        st.info("Run extraction first (Tab \u2460).")
    else:
        # Detect chemistry relevance
        _is_chem = is_chemistry_taxonomy(chem_taxonomy)
        if not _is_chem:
            st.warning(
                "The uploaded documents don't appear to be chemistry-related. "
                "The Chemistry Instructions view works best with laboratory / "
                "chemical procedure documents. You can still explore the view below."
            )

        chem_title = st.text_input(
            "Procedure Title",
            value="Chemistry Procedure",
            key="chem_title_input",
        )

        if st.button("Build Chemistry Instructions", use_container_width=True, key="btn_chem_build"):
            chem_instructions = build_chemistry_instructions(chem_taxonomy, chem_title)
            st.session_state["chem_instructions"] = chem_instructions
            st.rerun()

        chem_inst: ChemistryInstructionSet | None = st.session_state.get("chem_instructions")

        if chem_inst is not None:
            # ── Metrics row ──
            total_steps = len(chem_inst.steps)
            hazardous = sum(1 for s in chem_inst.steps if s.is_hazardous)
            pauses = len(chem_inst.safe_pause_indices)
            ts_count = len(chem_inst.troubleshooting)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Procedure Steps", total_steps)
            mc2.metric("Hazardous Steps", hazardous)
            mc3.metric("Safe Pause Points", pauses)
            mc4.metric("Troubleshooting Tips", ts_count)

            st.markdown("---")

            # ════════════════════════════════════
            # ZONE CARDS (The "Kit" Mental Model)
            # ════════════════════════════════════
            st.markdown("#### The Kit \u2014 Workspace Zones")
            st.markdown(
                "<p style='color:#6b7280; font-size:0.78rem;'>"
                "Before the first beaker is touched, review the entire landscape.</p>",
                unsafe_allow_html=True,
            )

            _zone_css = {
                KitZone.ZONE_A_SETUP: "chem-zone-a",
                KitZone.ZONE_B_REAGENTS: "chem-zone-b",
                KitZone.ZONE_C_REACTION: "chem-zone-c",
                KitZone.ZONE_D_WASTE: "chem-zone-d",
            }

            for zone in KitZone:
                zone_atoms = chem_inst.zones.get(zone, [])
                css_cls = _zone_css.get(zone, "")
                desc = ZONE_DESCRIPTIONS.get(zone, "")
                physical = ZONE_PHYSICAL.get(zone, "")
                icon_svg = ZONE_ICONS.get(zone, "")

                # --- Header: icon + label + descriptions ---
                zone_html = f"<div class='chem-zone-card {css_cls}'>"
                zone_html += "<div class='chem-zone-header'>"
                if icon_svg:
                    zone_html += f"<div class='chem-zone-icon'>{icon_svg}</div>"
                zone_html += "<div class='chem-zone-header-text'>"
                zone_html += f"<div class='chem-zone-label'>{html.escape(zone.value)}</div>"
                zone_html += f"<div class='chem-zone-desc'>{html.escape(desc)}</div>"
                if physical:
                    zone_html += (
                        f"<div class='chem-zone-physical'>"
                        f"\U0001f50d {html.escape(physical)}</div>"
                    )
                zone_html += "</div></div>"  # close header-text + header

                # --- Items table ---
                zone_html += "<div class='chem-zone-items'>"
                if zone_atoms:
                    zone_html += (
                        "<div class='chem-zone-items-header'>"
                        "<span class='chem-col-item'>Item</span>"
                        "<span class='chem-col-spec'>Specification</span>"
                        "<span class='chem-col-tag'>Status</span>"
                        "<span class='chem-col-src'>Source</span>"
                        "</div>"
                    )
                    for atom in zone_atoms:
                        prereq_badge = (
                            "<span class='badge-prereq'>PREREQ</span>"
                            if atom.is_prerequisite else
                            "<span style='color:#4a5568; font-size:0.68rem;'>\u2014</span>"
                        )
                        zone_html += (
                            f"<div class='atom-row'>"
                            f"<span class='atom-param'>{html.escape(atom.parameter)}</span>"
                            f"<span class='atom-value'>{html.escape(atom.value)}</span>"
                            f"<span class='atom-tag'>{prereq_badge}</span>"
                            f"<span class='atom-src'>{html.escape(atom.source_file)}</span>"
                            f"</div>"
                        )
                else:
                    zone_html += (
                        "<div style='color:#4a5568; font-size:0.78rem; "
                        "font-style:italic; padding: 0.75rem 0;'>"
                        "No items extracted for this zone.</div>"
                    )
                zone_html += "</div></div>"  # close items + card
                st.markdown(zone_html, unsafe_allow_html=True)

            st.markdown("---")

            # ════════════════════════════════════
            # ACTION-RESULT TABLE
            # ════════════════════════════════════
            st.markdown("#### Procedure Steps \u2014 Action \u2192 Expected Result")
            st.markdown(
                "<p style='color:#6b7280; font-size:0.78rem;'>"
                "Follow each step in order. Do NOT proceed if the expected "
                "result does not match.</p>",
                unsafe_allow_html=True,
            )

            if chem_inst.steps:
                for step in chem_inst.steps:
                    # Determine if this step needs a red box
                    if step.is_hazardous:
                        step_html = "<div class='red-box'>"
                        step_html += (
                            "<div class='red-box-label'>"
                            "HAZARDOUS \u2014 Read completely before acting"
                            "</div>"
                        )
                    else:
                        step_html = "<div style='margin-bottom:0.5rem;'>"

                    step_html += "<table class='ar-table'><tr>"
                    step_html += f"<td class='ar-step-num'>#{step.step_number}</td>"
                    step_html += f"<td class='ar-action'>{html.escape(step.action)}</td>"
                    step_html += f"<td class='ar-result'>{html.escape(step.expected_result)}</td>"
                    step_html += "</tr></table>"

                    # Safety note
                    if step.safety_note:
                        step_html += (
                            f"<div style='color:#fca5a5; font-size:0.75rem; "
                            f"padding-left:40px; margin-top:-0.3rem;'>"
                            f"Safety: {html.escape(step.safety_note)}</div>"
                        )

                    # Sensory checkpoints
                    if step.sensory_checkpoints:
                        _sensory_css = {
                            SensoryType.VISUAL: "sensory-visual",
                            SensoryType.AUDITORY: "sensory-auditory",
                            SensoryType.TACTILE: "sensory-tactile",
                            SensoryType.OLFACTORY: "sensory-olfactory",
                        }
                        _sensory_icons = {
                            SensoryType.VISUAL: "EYE",
                            SensoryType.AUDITORY: "EAR",
                            SensoryType.TACTILE: "HAND",
                            SensoryType.OLFACTORY: "NOSE",
                        }
                        for cp in step.sensory_checkpoints:
                            css = _sensory_css.get(cp.sensory_type, "")
                            icon = _sensory_icons.get(cp.sensory_type, "")
                            step_html += (
                                f"<span class='sensory-badge {css}'>"
                                f"{icon}: {html.escape(cp.description)}</span>"
                            )

                    step_html += "</div>"
                    st.markdown(step_html, unsafe_allow_html=True)

                    # Safe pause point marker
                    if step.is_safe_pause_point:
                        st.markdown(
                            "<div class='safe-pause'>"
                            "SAFE PAUSE POINT \u2014 chemicals are stable here; "
                            "you may step away</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.info(
                    "No procedural steps found. Ensure your source documents "
                    "contain step-by-step instructions."
                )

            st.markdown("---")

            # ════════════════════════════════════
            # TROUBLESHOOTING SIDEBAR
            # ════════════════════════════════════
            if chem_inst.troubleshooting:
                st.markdown("#### Troubleshooting Sidebars")
                st.markdown(
                    "<p style='color:#6b7280; font-size:0.78rem;'>"
                    "If something goes wrong, find the matching step below "
                    "before continuing.</p>",
                    unsafe_allow_html=True,
                )
                for ts in chem_inst.troubleshooting:
                    sev_css = (
                        "ts-card-critical" if ts.severity == "critical"
                        else ""
                    )
                    label_css = (
                        "ts-label-critical" if ts.severity == "critical"
                        else "ts-label-warning"
                    )
                    ts_html = f"<div class='ts-card {sev_css}'>"
                    ts_html += (
                        f"<div class='ts-label {label_css}'>"
                        f"{html.escape(ts.phase)} \u2014 "
                        f"{ts.severity.upper()}</div>"
                    )
                    ts_html += (
                        f"<div style='color:#e2e8f0;'>"
                        f"<strong>Symptom:</strong> "
                        f"{html.escape(ts.symptom)}</div>"
                    )
                    ts_html += (
                        f"<div style='color:#9ca3af;'>"
                        f"<strong>Cause:</strong> "
                        f"{html.escape(ts.probable_cause)}</div>"
                    )
                    ts_html += (
                        f"<div style='color:#86efac;'>"
                        f"<strong>Action:</strong> "
                        f"{html.escape(ts.corrective_action)}</div>"
                    )
                    ts_html += "</div>"
                    st.markdown(ts_html, unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 5 – LAYMAN DOC v3
# ════════════════════════════════════════════════

with tab_v3:
    st.markdown("### Layman Procedural Document v3")
    st.markdown(
        "<p style='color:#4a5568; font-size:0.82rem;'>"
        "Extracts a full structured procedure document conforming to "
        "layman-doc-schema-v3 — phases, sensory checkpoints, gates, "
        "emergency shutdown, materials acquisition cards, and more."
        "</p>",
        unsafe_allow_html=True,
    )

    v3_file_texts: dict = st.session_state.get("file_texts", {})

    if not v3_file_texts:
        st.info("Upload and extract documents first (Tab \u2460).")
    else:
        v3_title = st.text_input(
            "Procedure Title",
            value="Chemistry Procedure",
            key="v3_title_input",
        )
        v3_objective = st.text_input(
            "Procedure Objective (one sentence)",
            value="",
            key="v3_objective_input",
            placeholder="e.g. Synthesize aspirin from salicylic acid and acetic anhydride.",
        )

        # ── Pre-run cost / time estimate ──
        if v3_file_texts:
            est = estimate_pipeline_cost(
                v3_file_texts,
                provider=provider.lower(),
                tier=cost_tier,
                pipeline="layman_v3",
                model_override=model_override or None,
            )
            est_cols = st.columns(4)
            est_cols[0].metric("Est. API Calls", est["api_calls"])
            est_cols[1].metric("Est. Time", est["time_display"])
            est_cols[2].metric("Est. Cost", f"${est['cost_usd']:.3f}")
            est_cols[3].metric("Model", est["model"].split("-")[0].title())
            for warn in est.get("warnings", []):
                st.warning(warn)

        if st.button(
            "Generate Layman Doc v3",
            use_container_width=True,
            key="btn_v3_generate",
        ):
            if not api_key:
                st.error("Please enter your API key in the sidebar.")
            else:
                brain = LaymanV3Brain(
                    provider=provider.lower(),
                    api_key=api_key,
                    model=model_override or None,
                    tier=cost_tier,
                )

                n_files = len(v3_file_texts)
                progress = st.progress(0, text="Starting Layman v3 extraction\u2026")
                status_area = st.empty()

                def _v3_progress(idx: int, total: int, fname: str):
                    pct = int((idx / total) * 100)
                    progress.progress(pct, text=f"[{idx + 1}/{total}] Processing `{fname}`\u2026")
                    status_area.markdown(
                        f"<small style='color:#4a90d9;'>Extracting v3 from {fname}\u2026 "
                        f"(this may take a few minutes for large files)</small>",
                        unsafe_allow_html=True,
                    )

                try:
                    raw_doc = brain.extract_multi_documents(
                        v3_file_texts,
                        progress_callback=_v3_progress,
                    )
                    progress.progress(100, text="Extraction complete.")
                    status_area.markdown(
                        "<small style='color:#6ee7b7;'>\u2713 Extraction complete</small>",
                        unsafe_allow_html=True,
                    )

                    # Inject user-provided title/objective if supplied
                    if v3_title:
                        raw_doc.setdefault("meta", {})["title"] = v3_title
                    if v3_objective:
                        raw_doc.setdefault("meta", {})["objective"] = v3_objective
                    report = validate_v3_document(raw_doc)
                    fixed_doc = fix_v3_document(raw_doc, report)
                    # Re-validate after fix
                    report = validate_v3_document(fixed_doc)
                    st.session_state["v3_document"] = fixed_doc
                    st.session_state["v3_validation_report"] = report
                    st.rerun()
                except Exception as exc:
                    progress.progress(100, text="Extraction failed.")
                    st.error(f"Extraction failed: {exc}")

        # ── Display results ──
        v3_doc: dict | None = st.session_state.get("v3_document")
        v3_report: ValidationReport | None = st.session_state.get("v3_validation_report")

        if v3_doc is not None and v3_report is not None:
            # Metrics row
            stats = v3_report.stats
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Phases", stats.get("phases", 0))
            mc2.metric("Steps", stats.get("steps", 0))
            mc3.metric("Materials", stats.get("materials", 0))
            mc4.metric("Warnings", stats.get("warnings", 0))
            mc5.metric("Flags", len(v3_report.flags))

            st.markdown("---")

            # Validation report
            if v3_report.errors:
                with st.expander(
                    f"Validation Errors ({len(v3_report.errors)})", expanded=True
                ):
                    for err in v3_report.errors:
                        st.error(err)

            if v3_report.warnings:
                with st.expander(
                    f"Validation Warnings ({len(v3_report.warnings)})"
                ):
                    for warn in v3_report.warnings:
                        st.warning(warn)

            if v3_report.flags:
                with st.expander(f"Flagged Missing Data ({len(v3_report.flags)})"):
                    for flag in v3_report.flags:
                        st.caption(flag)

            st.markdown("---")

            # ── Section viewers ──

            # Meta
            meta = v3_doc.get("meta", {})
            with st.expander("Meta", expanded=True):
                st.markdown(f"**Title:** {meta.get('title', '')}")
                st.markdown(f"**Objective:** {meta.get('objective', '')}")
                st.markdown(
                    f"**Estimated Total Time:** {meta.get('estimatedTotalTime', 'N/A')}"
                )
                risks = meta.get("crossCuttingRisks", [])
                if risks:
                    st.markdown(f"**Cross-cutting Risks:** {len(risks)}")
                    for risk in risks:
                        st.markdown(
                            f"- **{risk.get('name', '')}**: {risk.get('risk', '')}"
                        )

            # Acquire — Materials
            materials = v3_doc.get("acquire", {}).get("materials", [])
            with st.expander(f"Materials ({len(materials)})"):
                for mat in materials:
                    st.markdown(
                        f"**{mat.get('commonName', '')}** "
                        f"({mat.get('quantity', '')}) "
                        f"— {mat.get('sourceType', '')}"
                    )
                    vis = mat.get("visualId", {})
                    if vis.get("appearance"):
                        st.caption(f"Appearance: {vis['appearance']}")

            # Stage — Zones
            zones = v3_doc.get("stage", {}).get("zones", [])
            with st.expander(f"Workspace Zones ({len(zones)})"):
                for zone in zones:
                    st.markdown(
                        f"**{zone.get('name', '')}** — {zone.get('description', '')}"
                    )

            # Execute — Phases & Steps
            phases = v3_doc.get("execute", {}).get("phases", [])
            for phase in phases:
                danger = phase.get("dangerLevel", "safe")
                _danger_color = {
                    "safe": "#22c55e", "caution": "#eab308", "dangerous": "#ef4444"
                }.get(danger, "#6b7280")
                phase_label = (
                    f"{phase.get('name', 'Phase')} "
                    f"({len(phase.get('steps', []))} steps)"
                )
                with st.expander(phase_label):
                    st.markdown(
                        f"<span style='color:{_danger_color}; font-weight:bold;'>"
                        f"{danger.upper()}</span> &nbsp; "
                        f"{phase.get('summary', '')}",
                        unsafe_allow_html=True,
                    )
                    ppe = phase.get("ppeForPhase", [])
                    if ppe:
                        st.markdown(f"**PPE:** {', '.join(ppe)}")

                    for step in phase.get("steps", []):
                        step_html = "<div style='margin:0.5rem 0; padding:0.5rem; "
                        step_html += "border-left:3px solid #3b82f6; padding-left:1rem;'>"
                        step_html += (
                            f"<strong>{html.escape(step.get('id', ''))}</strong> &nbsp; "
                            f"{html.escape(step.get('action', ''))}"
                        )
                        dod = step.get("directionOfDirection")
                        if dod:
                            step_html += (
                                f"<br><span style='color:#9ca3af; font-size:0.8rem;'>"
                                f"HOW: {html.escape(dod)}</span>"
                            )
                        sc = step.get("sensoryCheckpoint", {})
                        right = sc.get("right", {})
                        wrong = sc.get("wrong", {})
                        if right.get("description"):
                            step_html += (
                                f"<br><span style='color:#86efac; font-size:0.8rem;'>"
                                f"RIGHT: {html.escape(right['description'])}</span>"
                            )
                        if wrong.get("description"):
                            step_html += (
                                f"<br><span style='color:#fca5a5; font-size:0.8rem;'>"
                                f"WRONG: {html.escape(wrong['description'])}</span>"
                            )
                        for alert in step.get("alerts", []):
                            _alert_color = {
                                "red": "#ef4444", "yellow": "#eab308", "green": "#22c55e"
                            }.get(alert.get("level", ""), "#6b7280")
                            step_html += (
                                f"<br><span style='color:{_alert_color}; "
                                f"font-weight:bold; font-size:0.8rem;'>"
                                f"ALERT: {html.escape(alert.get('message', ''))}"
                                f"</span>"
                            )
                        step_html += "</div>"
                        st.markdown(step_html, unsafe_allow_html=True)

            # Verify
            tests = v3_doc.get("verify", {}).get("tests", [])
            with st.expander(f"Verification Tests ({len(tests)})"):
                for test in tests:
                    st.markdown(
                        f"**{test.get('name', '')}**: {test.get('method', '')}"
                    )

            # Shutdown
            sd_steps = v3_doc.get("shutdown", {}).get("steps", [])
            with st.expander(f"Shutdown ({len(sd_steps)} steps)"):
                for sd in sd_steps:
                    st.markdown(f"- {sd.get('action', '')}")

            # Emergency Shutdown
            esd = v3_doc.get("emergencyShutdown", {})
            esd_steps = esd.get("steps", [])
            with st.expander(
                f"Emergency Shutdown ({len(esd_steps)} steps)", expanded=False
            ):
                triggers = esd.get("triggerConditions", [])
                if triggers:
                    st.markdown("**Triggers:** " + ", ".join(triggers))
                for es in esd_steps:
                    st.markdown(
                        f"**{es.get('verb', '')}** {es.get('target', '')} "
                        f"— {es.get('detail', '')}"
                    )

            st.markdown("---")

            # Download JSON
            import json as _json

            v3_json_str = _json.dumps(v3_doc, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download v3 JSON",
                data=v3_json_str.encode("utf-8"),
                file_name=f"layman_v3_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )

# ════════════════════════════════════════════════
# TAB 6 – GAP AUDIT
# ════════════════════════════════════════════════

with tab_gap:
    st.markdown("### Step 5 \u00b7 Gap Audit Dashboard")

    taxonomy_for_gap = st.session_state.get("resolved_taxonomy") or st.session_state.get("taxonomy")

    if not taxonomy_for_gap:
        st.info("Run extraction first (Tab \u2460).")
    else:
        render_gap_audit(taxonomy_for_gap)

# ════════════════════════════════════════════════
# TAB 6 – EXPORT
# ════════════════════════════════════════════════

with tab_export:
    st.markdown("### Step 6 \u00b7 Export Final MPR")

    taxonomy_for_export = st.session_state.get("resolved_taxonomy") or st.session_state.get("taxonomy")

    if not taxonomy_for_export:
        st.info("Complete extraction (Tab \u2460) before exporting.")
    else:
        rs = st.session_state.get("resolution_state")
        if rs and rs.total_count > 0 and not st.session_state.get("resolution_done"):
            st.warning(
                "\u26a0\ufe0f Exporting pre-resolution document. "
                "For the final authoritative MPR, resolve conflicts in Tab \u2461 first."
            )

        doc_title = st.text_input(
            "Document Title",
            value="Master Production Record",
            help="This title appears at the top of the exported document.",
        )

        col_md, col_pdf = st.columns(2)

        with col_md:
            st.markdown("#### \U0001f4c4 Markdown Export")
            if st.button("Generate Markdown", use_container_width=True):
                md_content = build_markdown(taxonomy_for_export, doc_title)
                st.download_button(
                    label="\u2b07\ufe0f Download .md",
                    data=md_content.encode("utf-8"),
                    file_name=f"MPR_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
                with st.expander("Preview Markdown"):
                    st.code(md_content[:3000] + ("\u2026" if len(md_content) > 3000 else ""), language="markdown")

        with col_pdf:
            st.markdown("#### \U0001f4cb PDF Export")
            if st.button("Generate PDF", use_container_width=True):
                with st.spinner("Rendering PDF\u2026"):
                    pdf_bytes = build_pdf(taxonomy_for_export, doc_title)
                st.download_button(
                    label="\u2b07\ufe0f Download .pdf",
                    data=pdf_bytes,
                    file_name=f"MPR_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success("PDF ready for download.")
