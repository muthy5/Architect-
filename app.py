"""
app.py  –  The Operational Architect
Master Production Record Generator
"""

from __future__ import annotations

import html
import io
import textwrap
from datetime import datetime, timezone
from typing import Dict, List, Optional

import streamlit as st

from engine import (
    COST_TIERS,
    TAXONOMY_SECTIONS,
    OperationalBrain,
    TechnicalAtom,
    usage_stats,
)
from resolver import (
    Conflict,
    ResolutionState,
    apply_resolutions,
    detect_conflicts,
    flatten_taxonomy,
    resolve_conflict,
)
from self_healer import get_healer

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
        align-items: flex-start;
        padding: 0.35rem 0;
        border-bottom: 1px solid #1a1e2a;
        font-size: 0.82rem;
    }
    .atom-row:last-child { border-bottom: none; }
    .atom-param { color: #7dd3fc; min-width: 220px; font-weight: 600; }
    .atom-value { color: #ffffff; flex: 1; }
    .atom-src   { color: #4a5568; font-size: 0.72rem; white-space: nowrap; }
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
        for k, v in _fresh_defaults().items():
            st.session_state[k] = v
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
            except Exception:
                pass
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
        f'  <span class="atom-value">{value} {badges}</span>'
        f'  <span class="atom-src">{src}</span>'
        f'</div>'
    )


def render_taxonomy(taxonomy: Dict[str, List[TechnicalAtom]]) -> None:
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
                html = '<div class="mpr-card">'
                for atom in atoms:
                    html += _atom_html(atom)
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)


def render_gap_audit(taxonomy: Dict[str, List[TechnicalAtom]]) -> None:
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

def build_markdown(taxonomy: Dict[str, List[TechnicalAtom]], title: str) -> str:
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
            for atom in atoms:
                prereq = " `[PREREQ]`" if atom.is_prerequisite else ""
                hoisted = " `[HOISTED]`" if atom.hoisted else ""
                lines.append(
                    f"- **{atom.parameter}**: {atom.value}{prereq}{hoisted}  "
                    f"*(source: {atom.source_file})*"
                )
        lines.append("")
    return "\n".join(lines)


def build_pdf(taxonomy: Dict[str, List[TechnicalAtom]], title: str) -> bytes:
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
                for atom in atoms:
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
        # Self-healing: record failure and apply learned fallback
        healer = get_healer()
        healer.record_failure("pdf_export", e, context={"title": title})
        healer.learn_strategy("pdf_export", e, "fallback_pdf_to_markdown", succeeded=False)
        st.warning(f"PDF generation failed ({e}). Falling back to Markdown bytes.")
        md = build_markdown(taxonomy, title)
        return md.encode("utf-8")


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

tab_extract, tab_resolve, tab_mpr, tab_gap, tab_export = st.tabs(
    ["\u2460 Extract", "\u2461 Resolve Conflicts", "\u2462 MPR View", "\u2463 Gap Audit", "\u2464 Export"]
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
            st.markdown(f"- `{f.name}` ({f.size:,} bytes)")

        run_extraction = st.button("\U0001f52c Run Forensic Extraction", use_container_width=True)

        if run_extraction:
            if not api_key:
                st.error("\u26d4 No API key configured. Add your key in the sidebar.")
            elif not _key_ok:
                st.error("\u26d4 API key looks invalid for the selected provider. Fix it in the sidebar.")
            else:
                st.session_state["extraction_done"] = False
                st.session_state["resolution_done"] = False
                all_atoms: List[TechnicalAtom] = []
                file_texts: Dict[str, str] = {}

                brain = OperationalBrain(
                    provider=provider,
                    api_key=api_key,
                    model=model_override or None,
                    tier=cost_tier,
                )

                n_files = len(uploaded_files)
                progress = st.progress(0, text="Starting extraction\u2026")
                status_area = st.empty()
                failed_files: List[str] = []

                healer = get_healer()

                for i, uf in enumerate(uploaded_files):
                    # Progress: show file N/M and a sub-step
                    base_pct = int((i / n_files) * 100)
                    progress.progress(base_pct, text=f"[{i+1}/{n_files}] Reading `{uf.name}`\u2026")
                    status_area.markdown(
                        f"<small style='color:#4a90d9;'>Reading {uf.name}\u2026</small>",
                        unsafe_allow_html=True,
                    )

                    max_file_retries = 2
                    succeeded = False

                    for attempt in range(max_file_retries + 1):
                        try:
                            uf.seek(0)  # Reset file pointer for retries
                            text = read_file(uf)
                            if text.startswith("[PDF read error"):
                                st.warning(f"Could not read `{uf.name}`. It may be a scanned/image-only PDF.")
                                failed_files.append(uf.name)
                                break
                            if len(text.strip()) < 20:
                                st.warning(f"`{uf.name}` appears to be empty or too short to extract from.")
                                failed_files.append(uf.name)
                                break
                            file_texts[uf.name] = text

                            # Sub-step: extraction via LLM
                            mid_pct = int(((i + 0.5) / n_files) * 100)
                            progress.progress(mid_pct, text=f"[{i+1}/{n_files}] Extracting specs from `{uf.name}`\u2026")
                            status_area.markdown(
                                f"<small style='color:#4a90d9;'>Calling LLM for {uf.name}\u2026 (this may take a moment)</small>",
                                unsafe_allow_html=True,
                            )

                            atoms = brain.extract_atoms(text, uf.name)
                            all_atoms.extend(atoms)

                            recovery_note = ""
                            if attempt > 0:
                                recovery_note = f" (self-healed after {attempt} retry)"
                                healer.record_success(
                                    "file_extraction",
                                    context={"filename": uf.name, "attempt": attempt},
                                    recovery_strategy="retry_with_backoff",
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

                taxonomy = brain.organize_by_taxonomy(all_atoms)
                resolution_state = detect_conflicts(all_atoms)

                st.session_state["raw_atoms"] = all_atoms
                st.session_state["taxonomy"] = taxonomy
                st.session_state["file_texts"] = file_texts
                st.session_state["resolution_state"] = resolution_state
                st.session_state["extraction_done"] = True

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
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Atoms Extracted", total)
            c2.metric("Files Processed", len(st.session_state["file_texts"]))
            c3.metric("Conflicts Detected", conflicts, delta=None)

            if conflicts > 0:
                st.warning(
                    f"\u26a0\ufe0f {conflicts} conflict(s) detected. "
                    "Switch to the **\u2461 Resolve Conflicts** tab to proceed."
                )
            else:
                st.success("\u2705 No conflicts. Navigate to **\u2462 MPR View** or **\u2464 Export**.")

# ════════════════════════════════════════════════
# TAB 2 – CONFLICT RESOLUTION
# ════════════════════════════════════════════════

with tab_resolve:
    st.markdown("### Step 2 \u00b7 Resolve Data Conflicts")

    rs: Optional[ResolutionState] = st.session_state.get("resolution_state")

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
                st.success("Resolutions applied. Navigate to **\u2462 MPR View** or **\u2464 Export**.")
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
# TAB 4 – GAP AUDIT
# ════════════════════════════════════════════════

with tab_gap:
    st.markdown("### Step 4 \u00b7 Gap Audit Dashboard")

    taxonomy_for_gap = st.session_state.get("resolved_taxonomy") or st.session_state.get("taxonomy")

    if not taxonomy_for_gap:
        st.info("Run extraction first (Tab \u2460).")
    else:
        render_gap_audit(taxonomy_for_gap)

# ════════════════════════════════════════════════
# TAB 5 – EXPORT
# ════════════════════════════════════════════════

with tab_export:
    st.markdown("### Step 5 \u00b7 Export Final MPR")

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
