"""Page stylesheet: colours, cards and animations.

Only classes prefixed with `osc-` (our own HTML) are styled, plus a few stable
Streamlit hooks (`data-testid`), so a Streamlit upgrade is unlikely to break the look.
Every animation is disabled for users who ask their OS for reduced motion.
"""

# Colour tokens (validated categorical palette: blue = primary, orange = purchase/accent)
PRIMARY = "#2a78d6"
PRIMARY_DARK = "#1d5fae"
ACCENT = "#eb6834"
GOOD = "#1baf7a"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
BORDER = "#e6e6e3"
SURFACE = "#ffffff"

# Performance bands use the reserved status colours, always paired with a symbol + label
BAND_GOOD = "#0ca30c"
BAND_MODERATE = "#fab219"
BAND_LOW = "#d03b3b"
NEUTRAL = "#b9b8b2"

_CSS = f"""
<style>
/* ---------- animations ---------- */
@keyframes osc-fade-up {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes osc-pop {{ from {{ opacity: 0; transform: scale(.94); }} to {{ opacity: 1; transform: scale(1); }} }}
@keyframes osc-grow {{ from {{ width: 0; }} }}
@keyframes osc-fade-in {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes osc-gradient {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
@keyframes osc-pulse {{
  0%, 100% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(42,120,214,.35); }}
  50% {{ transform: scale(1.06); box-shadow: 0 0 0 16px rgba(42,120,214,0); }}
}}

/* ---------- layout polish ---------- */
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }}
[data-testid="stFileUploaderDropzone"] {{ border: 2px dashed {PRIMARY}55; border-radius: 12px; transition: background .2s ease; }}
[data-testid="stFileUploaderDropzone"]:hover {{ background: {PRIMARY}0d; }}

/* ---------- section jump links ---------- */
.osc-nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 4px; animation: osc-fade-up .5s ease both; }}
.osc-nav a {{ text-decoration: none; padding: 6px 14px; border-radius: 999px; background: {SURFACE}; border: 1px solid {BORDER};
  color: {TEXT} !important; font-size: .85rem; font-weight: 600; transition: all .15s ease; }}
.osc-nav a:hover {{ border-color: {PRIMARY}; color: {PRIMARY_DARK} !important; background: {PRIMARY}0d; }}

/* ---------- hero banner ---------- */
.osc-hero {{
  background: linear-gradient(120deg, #12355f, {PRIMARY_DARK}, {PRIMARY}, #3b4fb8);
  background-size: 300% 300%;
  animation: osc-gradient 16s ease infinite, osc-fade-up .6s ease both;
  color: #fff; border-radius: 18px; padding: 26px 30px; margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(18,53,95,.18);
}}
.osc-hero h1 {{ color: #fff; margin: 0 0 6px 0; padding: 0; font-size: 2.05rem; font-weight: 750; letter-spacing: -.01em; }}
.osc-hero p {{ color: rgba(255,255,255,.92); margin: 0 0 12px 0; font-size: 1.02rem; max-width: 760px; }}
.osc-chip {{ display: inline-block; padding: 4px 11px; margin: 4px 6px 0 0; border-radius: 999px;
  background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); font-size: .8rem; color: #fff; }}

/* ---------- author credit: header byline + corner watermark ---------- */
.osc-byline {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; width: fit-content; margin-top: 14px;
  padding: 6px 13px; border-radius: 10px; background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.3);
  color: #fff; font-size: .88rem; }}
.osc-byline b {{ color: #fff; }}
.osc-sep {{ opacity: .6; }}
.osc-watermark {{ position: fixed; right: 18px; bottom: 14px; z-index: 1000; pointer-events: none;
  font-size: .74rem; font-weight: 600; color: {TEXT_SECONDARY}; letter-spacing: .01em;
  background: rgba(255,255,255,.82); border: 1px solid {BORDER}; border-radius: 999px; padding: 5px 13px;
  box-shadow: 0 2px 10px rgba(0,0,0,.06); backdrop-filter: blur(4px); opacity: .85; }}

/* ---------- generic card ---------- */
.osc-card {{
  background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 16px 18px; box-shadow: 0 1px 2px rgba(0,0,0,.04);
  animation: osc-fade-up .5s ease both; transition: transform .18s ease, box-shadow .18s ease;
}}
.osc-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 22px rgba(0,0,0,.08); }}
.osc-card h4 {{ margin: 0 0 4px 0; padding: 0; font-size: 1rem; color: {TEXT}; }}
.osc-card p {{ margin: 0; color: {TEXT_SECONDARY}; font-size: .9rem; line-height: 1.45; }}
.osc-muted {{ color: {TEXT_MUTED}; font-size: .8rem; }}

/* ---------- grids ---------- */
.osc-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin: 6px 0 14px 0; }}
.osc-grid-3 {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin: 8px 0 16px 0; }}

/* ---------- step cards ---------- */
.osc-step-num {{ display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px;
  border-radius: 50%; background: {PRIMARY}; color: #fff; font-size: .8rem; font-weight: 700; margin-right: 8px; }}
.osc-step-icon {{ font-size: 1.6rem; float: right; opacity: .85; }}

/* ---------- empty state ---------- */
.osc-empty {{ text-align: center; padding: 34px 20px 26px; border: 2px dashed {PRIMARY}44; border-radius: 18px;
  background: linear-gradient(180deg, {PRIMARY}08, transparent); animation: osc-fade-up .5s ease both; margin-bottom: 10px; }}
.osc-empty-icon {{ width: 78px; height: 78px; margin: 0 auto 12px; border-radius: 50%; background: {PRIMARY}14;
  display: flex; align-items: center; justify-content: center; font-size: 2.2rem; animation: osc-pulse 2.4s ease-in-out infinite; }}
.osc-empty h3 {{ margin: 0 0 6px 0; color: {TEXT}; }}
.osc-empty p {{ margin: 0 auto; color: {TEXT_SECONDARY}; max-width: 560px; }}

/* ---------- metric cards ---------- */
.osc-metric {{ cursor: help; }}
.osc-metric .osc-label {{ font-size: .74rem; font-weight: 650; color: {TEXT_SECONDARY}; text-transform: uppercase; letter-spacing: .06em; }}
.osc-metric .osc-value {{ font-size: 1.85rem; font-weight: 750; color: {TEXT}; margin: 2px 0 8px; font-variant-numeric: tabular-nums; }}
.osc-metric .osc-best {{ float: right; font-size: .7rem; color: {PRIMARY_DARK}; background: {PRIMARY}14; padding: 2px 7px; border-radius: 999px; }}
.osc-bar {{ height: 7px; background: #eef1f5; border-radius: 99px; overflow: hidden; }}
.osc-bar > span {{ display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, {PRIMARY_DARK}, {PRIMARY});
  animation: osc-grow 1.1s cubic-bezier(.2,.8,.2,1) both; }}
.osc-hint {{ font-size: .78rem; color: {TEXT_MUTED}; margin-top: 7px; }}

/* ---------- badges & callouts ---------- */
.osc-badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .76rem; font-weight: 600; margin: 2px 6px 2px 0; }}
.osc-badge.blue {{ background: {PRIMARY}17; color: {PRIMARY_DARK}; }}
.osc-badge.green {{ background: {GOOD}1c; color: #0b7a52; }}
.osc-badge.orange {{ background: {ACCENT}1c; color: #b4451a; }}
.osc-callout {{ border-left: 4px solid {PRIMARY}; background: {PRIMARY}0b; padding: 10px 14px; border-radius: 8px;
  color: {TEXT}; margin: 4px 0 12px; animation: osc-fade-up .5s ease both; }}

/* ---------- confusion matrix ---------- */
.osc-cm {{ display: grid; grid-template-columns: 118px 1fr 1fr; gap: 8px; align-items: stretch; }}
.osc-cm-head, .osc-cm-side {{ font-size: .78rem; color: {TEXT_SECONDARY}; display: flex; flex-direction: column; justify-content: center; }}
.osc-cm-head {{ text-align: center; padding-bottom: 2px; }}
.osc-cm-head b, .osc-cm-side b {{ color: {TEXT}; font-size: .86rem; }}
.osc-cm-cell {{ border-radius: 12px; padding: 14px 10px; text-align: center; animation: osc-pop .45s ease both; border: 1px solid transparent; }}
.osc-cm-cell .osc-count {{ font-size: 1.7rem; font-weight: 750; font-variant-numeric: tabular-nums; }}
.osc-cm-cell .osc-share {{ font-size: .78rem; opacity: .85; }}
.osc-cm-cell .osc-tag {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; margin-top: 4px; font-weight: 650; }}
.osc-cm-cell.correct {{ background: {PRIMARY}16; color: #123f73; border-color: {PRIMARY}33; }}
.osc-cm-cell.wrong {{ background: {ACCENT}14; color: #8a3410; border-color: {ACCENT}33; }}

/* ---------- model card (setup panel) ---------- */
.osc-model h4 {{ display: flex; justify-content: space-between; align-items: center; }}
.osc-param {{ display: inline-block; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .74rem;
  background: #f3f4f6; color: {TEXT}; border-radius: 6px; padding: 2px 6px; margin: 3px 4px 0 0; }}

/* ---------- performance bands (colour + symbol + label, never colour alone) ---------- */
.osc-band {{ display: inline-flex; align-items: center; gap: 5px; font-size: .72rem; font-weight: 650; color: {TEXT_SECONDARY}; }}
.osc-dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none; }}
.osc-good {{ --band: {BAND_GOOD}; }}
.osc-moderate {{ --band: {BAND_MODERATE}; }}
.osc-low {{ --band: {BAND_LOW}; }}
.osc-dot, .osc-fill {{ background: var(--band, {PRIMARY}); }}
.osc-metric.osc-good, .osc-metric.osc-moderate, .osc-metric.osc-low {{ border-left: 5px solid var(--band); }}
.osc-metric .osc-bar > span {{ background: var(--band, {PRIMARY}); }}
.osc-legend {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin: 2px 0 10px; font-size: .78rem; color: {TEXT_SECONDARY}; }}

/* ---------- hover tooltips (pure CSS) ---------- */
.osc-tip {{ position: relative; }}
.osc-tooltip {{ visibility: hidden; opacity: 0; position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%);
  background: #1f1f1e; color: #fff; padding: 6px 9px; border-radius: 7px; font-size: .74rem; line-height: 1.35; white-space: nowrap;
  pointer-events: none; z-index: 20; transition: opacity .15s ease; box-shadow: 0 4px 14px rgba(0,0,0,.18); }}
.osc-tip:hover .osc-tooltip {{ visibility: visible; opacity: 1; }}

/* ---------- position chart: where the selected model sits among all five ---------- */
.osc-pos-row {{ display: grid; grid-template-columns: 92px 1fr 74px; align-items: center; gap: 12px; padding: 7px 0; }}
.osc-pos-row + .osc-pos-row {{ border-top: 1px solid #f0f0ee; }}
.osc-pos-name {{ font-size: .82rem; font-weight: 650; color: {TEXT}; }}
.osc-pos-value {{ font-size: .86rem; font-weight: 700; color: {TEXT}; text-align: right; font-variant-numeric: tabular-nums; }}
.osc-track {{ position: relative; height: 22px; background: repeating-linear-gradient(90deg, transparent 0 calc(25% - 1px), #ececea calc(25% - 1px) 25%); }}
.osc-track::before {{ content: ""; position: absolute; left: 0; right: 0; top: 50%; height: 2px; background: #e4e4e1; transform: translateY(-50%); }}
.osc-range {{ position: absolute; top: 50%; height: 8px; transform: translateY(-50%); background: #e1e0db; border-radius: 4px; }}
.osc-track .osc-mark {{ position: absolute; top: 50%; transform: translate(-50%, -50%); border-radius: 50%; cursor: default;
  animation: osc-fade-in .6s ease both; }}
.osc-track .osc-mark.other {{ width: 10px; height: 10px; background: {NEUTRAL}; box-shadow: 0 0 0 2px {SURFACE}; }}
.osc-track .osc-mark.selected {{ width: 16px; height: 16px; background: var(--band, {PRIMARY});
  box-shadow: 0 0 0 3px {SURFACE}, 0 0 0 4px var(--band, {PRIMARY}); z-index: 2; }}
.osc-track .osc-mark:hover {{ filter: brightness(.85); z-index: 3; }}
.osc-axis {{ display: grid; grid-template-columns: 92px 1fr 74px; gap: 12px; font-size: .7rem; color: {TEXT_MUTED}; }}
.osc-axis-scale {{ display: flex; justify-content: space-between; }}

/* ---------- ranking bars ---------- */
.osc-rank-row {{ display: grid; grid-template-columns: 190px 1fr 70px; align-items: center; gap: 12px; padding: 5px 0; }}
.osc-rank-name {{ font-size: .86rem; color: {TEXT}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.osc-rank-row.selected .osc-rank-name {{ font-weight: 750; }}
.osc-rank-track {{ height: 18px; background: #f1f2f4; border-radius: 4px; }}
.osc-rank-track .osc-fill {{ height: 100%; border-radius: 4px; animation: osc-grow 1s cubic-bezier(.2,.8,.2,1) both; }}
.osc-rank-row.selected .osc-rank-track {{ outline: 2px solid {TEXT}; outline-offset: 2px; }}
.osc-rank-value {{ font-size: .86rem; font-weight: 700; text-align: right; color: {TEXT}; font-variant-numeric: tabular-nums; }}

/* ---------- outcome mix: stacked bar of TN / TP / FP / FN ---------- */
.osc-stack {{ display: flex; gap: 2px; height: 26px; margin: 8px 0 10px; }}
.osc-seg {{ height: 100%; animation: osc-grow 1s cubic-bezier(.2,.8,.2,1) both; }}
.osc-seg:first-child {{ border-radius: 4px 0 0 4px; }}
.osc-seg:last-child {{ border-radius: 0 4px 4px 0; }}
.osc-seg:hover {{ filter: brightness(1.08); }}

/* ---------- accessibility: respect reduced-motion preference ---------- */
@media (prefers-reduced-motion: reduce) {{
  .osc-hero, .osc-card, .osc-empty, .osc-empty-icon, .osc-bar > span, .osc-cm-cell, .osc-callout,
  .osc-mark, .osc-fill, .osc-seg {{ animation: none !important; }}
}}
</style>
"""


def page_css() -> str:
    """The complete <style> block injected once at the top of the app."""
    return _CSS
