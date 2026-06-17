"""Interactive HTML graph viewer (pyvis / vis.js).

Smooth, physics-based, floating nodes:
  - Drag any node — physics keeps the rest balanced
  - Mouse-wheel zoom, click-and-drag to pan
  - Hover any node to highlight its connections and see the tooltip
  - Sector-coloured nodes (12 sectors + 'other'); node size = weighted degree
  - Date selector dropdown: pick any test day, view that day's graph; the
    "Average graph" entry shows the time-averaged adjacency

Reads `<run>/graphs/test_*.npz` written during training.
Output: a single self-contained HTML at `<run>/figures/interactive_graph.html`
        (just open it in a browser).

Usage:
    python scripts/interactive_graph.py runs/v2_aapl
    python scripts/interactive_graph.py runs/v2_aapl --per-node-topk 5
    python scripts/interactive_graph.py runs/v2_aapl --date 2025-10-17
"""
from __future__ import annotations

import argparse
import glob
import json
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Sector palette  (color-blind friendly Tab10 + a neutral grey)
# ──────────────────────────────────────────────────────────────────────────
SECTOR_COLORS = {
    "tech":           "#4c78a8",
    "financials":     "#f58518",
    "health":         "#54a24b",
    "industrials":    "#b279a2",
    "discretionary":  "#e45756",
    "staples":        "#72b7b2",
    "comms":          "#eeca3b",
    "energy":         "#9d755d",
    "utilities":      "#bab0ab",
    "real_estate":    "#ff9da6",
    "materials":      "#5a5a5a",
    "other":          "#999999",
}


# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────
def _load_graphs(run: Path) -> List[dict]:
    files = sorted(glob.glob(str(run / "graphs" / "test_*.npz")))
    out = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        out.append({
            "date": str(d["date"]),
            "adj": d["adj"].astype(np.float32),
            "eps": float(d["eps"]),
            "tickers": d["tickers"].tolist(),
        })
    return out


def _target_ticker(run: Path) -> str:
    cfg = run / "config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text()).get("target_ticker", "TARGET").upper()
        except Exception:
            pass
    return "TARGET"


def _load_sectors(run: Path, tickers: List[str]) -> Dict[str, str]:
    import pandas as pd
    candidates = [
        run.parent.parent / "data" / "sectors_static_labels.csv",
        Path("data") / "sectors_static_labels.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            mapping = dict(zip(df["ticker"].astype(str), df["sector"].astype(str)))
            return {t: mapping.get(t, "other") for t in tickers}
    return {t: "other" for t in tickers}


# ──────────────────────────────────────────────────────────────────────────
# Choose which adjacency to show
# ──────────────────────────────────────────────────────────────────────────
def _pick_adj(graphs: List[dict], date: Optional[str]) -> Tuple[np.ndarray, str, float]:
    """Return (adjacency, title, eps) — either a specific date or the time-mean."""
    if date is not None:
        for g in graphs:
            if g["date"] == date:
                return g["adj"], f"date: {date}", g["eps"]
        raise SystemExit(f"date {date} not in run; available: {[g['date'] for g in graphs][:5]}…")
    adj = np.mean([g["adj"] for g in graphs], axis=0)
    eps = float(np.mean([g["eps"] for g in graphs]))
    return adj, f"average over {len(graphs)} test days", eps


def _top_neighbours(adj_row: np.ndarray, tickers: List[str], k: int = 5) -> List[Tuple[str, float]]:
    idx = np.argsort(-adj_row)
    out: List[Tuple[str, float]] = []
    for j in idx:
        if adj_row[j] <= 1e-6:
            continue
        out.append((tickers[j], float(adj_row[j])))
        if len(out) >= k:
            break
    return out


def _compute_layout(A: np.ndarray, seed: int = 42, scale: float = 480.0) -> np.ndarray:
    """One fixed [N, 2] layout from the (already filtered) adjacency, so we can
    PIN node positions and run with physics DISABLED — stable, no rotation, no
    drift. Kamada-Kawai on the largest component (prettier for sparse graphs),
    falling back to spring; isolates ring around the outside."""
    import networkx as nx
    n = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(A[i, j])
            if w > 0:
                G.add_edge(i, j, weight=w)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    pos: dict = {}
    if comps and len(comps[0]) > 1:
        sub = G.subgraph(comps[0])
        try:
            pos = nx.kamada_kawai_layout(sub, weight="weight")
        except Exception:
            pos = nx.spring_layout(sub, seed=seed, weight="weight",
                                   k=2.5 / max(np.sqrt(len(sub)), 1.0), iterations=400)
    arr0 = np.array(list(pos.values())) if pos else np.zeros((0, 2))
    radius = (np.linalg.norm(arr0 - arr0.mean(0), axis=1).max() if len(arr0) else 1.0) or 1.0
    isolates = [i for i in range(n) if i not in pos]
    for k_, i in enumerate(isolates):
        ang = 2 * np.pi * k_ / max(len(isolates), 1)
        pos[i] = np.array([radius * 1.6 * np.cos(ang), radius * 1.6 * np.sin(ang)])
    arr = np.array([pos[i] for i in range(n)], dtype=np.float32)
    arr -= arr.mean(0)
    span = np.abs(arr).max() or 1.0
    return arr / span * scale


# ──────────────────────────────────────────────────────────────────────────
# Build the pyvis network
# ──────────────────────────────────────────────────────────────────────────
def build_network(
    adj: np.ndarray,
    tickers: List[str],
    sectors: Dict[str, str],
    target_ticker: str,
    per_node_topk: int = 5,           # show each node's top-k strongest edges
    title_extra: str = "",
):
    from pyvis.network import Network

    # symmetrise + zero the diagonal
    A = np.maximum(adj, adj.T).astype(np.float32)
    np.fill_diagonal(A, 0.0)

    # ── per-node top-k filter (NOT global cutoff) ────────────────────────
    # The old viz used a global percentile cutoff which hid the actual
    # edges the model learned for peripheral stocks. Now we mirror what
    # the model itself does: keep the top-k strongest edges per node.
    # Every node gets at least min(k, n-1) edges → no isolated nodes.
    if per_node_topk > 0 and per_node_topk < A.shape[0]:
        keep = np.zeros_like(A, dtype=bool)
        for i in range(A.shape[0]):
            # indices of the k strongest neighbours for row i
            top_idx = np.argpartition(-A[i], per_node_topk)[:per_node_topk]
            keep[i, top_idx] = True
        # symmetrise the mask — i↔j is shown if either direction made the cut
        keep = keep | keep.T
        A = np.where(keep, A, 0.0)

    # Degree = sum of edge weights per node.
    deg = A.sum(axis=1)
    max_deg = max(deg.max() if deg.size else 1.0, 1e-6)

    n = len(tickers)
    target_idx = tickers.index(target_ticker) if target_ticker in tickers else -1

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#15151c",
        font_color="#e6e6e6",
        notebook=False,
        cdn_resources="in_line",
        directed=False,
    )
    # Physics DISABLED. Node positions are precomputed once (see _compute_layout)
    # and PINNED, so the graph is completely stable — it never rotates, drifts,
    # or jitters, and there's nothing to "settle". You can still drag any node to
    # reposition it (it just stays where you drop it), scroll to zoom, and pan.
    # This is the most predictable way to "look around" a fixed graph.
    net.set_options("""{
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {"color": "#e6e6e6", "size": 12, "face": "Inter, system-ui"},
        "shape": "dot"
      },
      "edges": {
        "color": {"color": "rgba(255,255,255,0.18)", "highlight": "#ffd166"},
        "smooth": {"type": "continuous"},
        "width": 0.6
      },
      "physics": { "enabled": false },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "hideEdgesOnDrag": false,
        "navigationButtons": true,
        "keyboard": {"enabled": false},
        "dragNodes": true,
        "zoomView": true,
        "dragView": true
      }
    }""")

    # one fixed layout for the whole graph (physics is off, so we must place nodes)
    pos = _compute_layout(A, seed=42)

    # Add nodes.
    for i, t in enumerate(tickers):
        sector = sectors.get(t, "other")
        colour = SECTOR_COLORS.get(sector, SECTOR_COLORS["other"])
        size = float(12.0 + 30.0 * (float(deg[i]) / max_deg))
        is_target = (i == target_idx)

        # Tooltip: top-5 neighbours.
        top = _top_neighbours(A[i], tickers, k=5)
        if top:
            n_html = "<br/>".join(
                f"<span style='color:#ffd166;'>{t2}</span> · "
                f"<span style='color:#8be9fd;'>w={w:.3f}</span>"
                for t2, w in top
            )
        else:
            n_html = "<i>(no neighbours past cutoff)</i>"

        title_html = (
            f"<div style='font-family:Inter,system-ui;font-size:13px;"
            f"min-width:230px;padding:8px;background:#1f1f28;border-radius:8px;'>"
            f"<div style='font-size:15px;font-weight:600;color:{colour};margin-bottom:4px;'>"
            f"{t}{' ★ target' if is_target else ''}</div>"
            f"<div style='color:#999;font-size:11px;margin-bottom:6px;'>"
            f"sector: <b style='color:{colour};'>{sector}</b> · "
            f"weighted deg: {float(deg[i]):.2f}</div>"
            f"<div style='color:#ddd;font-size:11px;margin-bottom:2px;'>top neighbours:</div>"
            f"<div style='font-size:11px;line-height:1.55;'>{n_html}</div>"
            f"</div>"
        )

        label = t
        borderWidth = 4 if is_target else 2
        borderColor = "#ffd166" if is_target else colour

        net.add_node(
            i,
            label=label,
            title=title_html,
            color={"background": colour, "border": borderColor},
            size=size,
            borderWidth=borderWidth,
            x=float(pos[i, 0]), y=float(pos[i, 1]),   # pinned (physics off)
            physics=False,
        )

    # Add edges (above the cutoff).
    max_edge = float(A.max()) if A.max() > 0 else 1.0
    for i in range(n):
        for j in range(i + 1, n):
            w = float(A[i, j])
            if w <= 0:
                continue
            # Edge width scaled to weight; cap at 6 px.
            ew = float(0.4 + 5.6 * (w / max_edge))
            net.add_edge(i, j, value=float(w), width=ew, title=f"weight {w:.3f}")

    # Insert a small custom header above the canvas (sector legend).
    legend_items = "".join(
        f"<span style='display:inline-block;margin:0 14px 6px 0;font-size:12px;color:#bbb;'>"
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
        f"background:{c};margin-right:6px;vertical-align:middle;'></span>{s}</span>"
        for s, c in SECTOR_COLORS.items()
    )
    title = f"TAGC graph — target {target_ticker}  ·  {title_extra}"
    return net, title, legend_items


def _wrap_with_header(net_html: str, title: str, legend: str, run_path: Path,
                     date_options: List[str], current_date: Optional[str]) -> str:
    """Wrap the pyvis HTML with a header + date selector (if multiple snapshots)."""
    date_select = ""
    if len(date_options) > 1:
        opts = []
        opts.append(f'<option value="" {"selected" if current_date is None else ""}>'
                    f'Average over all test days ({len(date_options)})</option>')
        for d in date_options:
            sel = "selected" if d == current_date else ""
            opts.append(f'<option value="{d}" {sel}>{d}</option>')
        date_select = (
            '<div style="display:inline-flex;align-items:center;gap:10px;'
            'margin-left:24px;font-size:13px;color:#bbb;">'
            '<span>Snapshot:</span>'
            '<select id="date-picker" '
            'onchange="window.location.search=\'?date=\'+this.value"'
            'style="background:#1f1f28;color:#e6e6e6;border:1px solid #444;'
            'padding:6px 10px;border-radius:6px;font-size:13px;">'
            f'{"".join(opts)}'
            '</select></div>'
        )

    header = f"""
<div style="background:#0e0e14; color:#e6e6e6; padding:14px 22px;
            font-family:Inter,system-ui,-apple-system,sans-serif;
            border-bottom:1px solid #2a2a36;">
  <div style="display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
    <div style="font-size:16px; font-weight:600;">{title}</div>
    {date_select}
    <div style="margin-left:auto; font-size:11px; color:#888;">
      drag nodes · scroll to zoom · hover for neighbours · physics OFF (fixed layout, no drift)
    </div>
  </div>
  <div style="margin-top:8px;">{legend}</div>
</div>
<style>
  body {{ margin:0; background:#15151c; }}
  #mynetwork {{ background:#15151c !important; border:none !important; }}
  .vis-tooltip {{ background:#1f1f28 !important; border:1px solid #333 !important;
                  color:#e6e6e6 !important; padding:0 !important; }}
  .vis-navigation .vis-button {{ background-color:#2a2a36 !important;
                                  filter: invert(0.85); }}
</style>
"""
    # Drop pyvis's own header / nav, then prepend ours.
    # pyvis output starts with <html>...<body>...; we splice the header right after <body>.
    if "<body>" in net_html:
        net_html = net_html.replace("<body>", "<body>" + header, 1)
    return net_html


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--per-node-topk", type=int, default=5,
                    help="show each node's top-k strongest edges (default 5). "
                         "0 = show all edges. This mirrors the model's own top-k "
                         "rule so every node is guaranteed at least 1 connection.")
    ap.add_argument("--date", default=None,
                    help="specific test day to render; default = average")
    ap.add_argument("--out", default=None,
                    help="output html path; default = <run>/figures/interactive_graph.html")
    ap.add_argument("--open", action="store_true",
                    help="open the produced file in the default browser")
    args = ap.parse_args()

    run = Path(args.run_dir)
    graphs = _load_graphs(run)
    if not graphs:
        raise SystemExit(f"no graphs in {run}/graphs/test_*.npz — train first")

    target = _target_ticker(run)
    tickers = graphs[0]["tickers"]
    sectors = _load_sectors(run, tickers)
    date_options = [g["date"] for g in graphs]

    adj, label, eps = _pick_adj(graphs, args.date)
    n_sectors_present = len({sectors[t] for t in tickers})
    print(f"target  = {target}")
    print(f"frames  = {len(graphs)} ({date_options[0]} → {date_options[-1]})")
    print(f"sectors = {n_sectors_present} present")
    print(f"showing = {label}   ε = {eps:.3f}")

    title_extra = (f"{label} · ε={eps:.3f} · "
                    f"top-{args.per_node_topk} edges per node "
                    f"(matches the model's own sparsification)")
    net, title, legend = build_network(adj, tickers, sectors, target,
                                        per_node_topk=args.per_node_topk,
                                        title_extra=title_extra)

    out = Path(args.out) if args.out else run / "figures" / "interactive_graph.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    # pyvis writes the file itself; we wrap it after with our header.
    raw_html = net.generate_html(notebook=False)
    full_html = _wrap_with_header(raw_html, title, legend, run, date_options, args.date)
    out.write_text(full_html, encoding="utf-8")

    size_mb = out.stat().st_size / 1e6
    print(f"wrote   {out}  ({size_mb:.1f} MB)")
    print(f"open    file://{out.resolve()}")
    if args.open:
        webbrowser.open(f"file://{out.resolve()}")


if __name__ == "__main__":
    main()
