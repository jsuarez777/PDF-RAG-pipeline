#!/usr/bin/env python3
"""Visualize retrieval-evaluation results across the experiment grid.

Reads the eval_*.json files written by eval_retrieval.py (one per
chunk-config x index combination) and renders the project's required charts
to visualizations/<ts>/:

  1. mrr          - MRR comparison bar chart, colored by retrieval method
  2. scatter      - Recall@K vs Precision@K scatter, top-5 experiments labeled
  3. heatmap      - chunk config x index heatmap of the chosen metric
  4. retrieval    - grouped bar: metric per chunk config, one bar per index
  5. correlation  - correlation matrix of all IR metrics across experiments
  6. recall-curve - Recall@K vs K line chart, small multiples per chunk config
  7. qtype        - metric by question type (direct/inference/paraphrased)
                    per experiment; extra data eval_retrieval.py collects

The project also asks for a response-time vs quality scatter; eval_retrieval.py
does not record retrieval timing yet, so that chart is not implemented.

Evals are computed at ks {1,3,5,10}; the project mostly reports @5, so
metric/k-dependent charts take --metric and -k (default mrr / k=5).
By default the latest eval per (chunk config, index) is used; --all-evals
keeps re-runs of the same combination apart.

Run from the project root:

    python app/generate_visualizations.py                    # all charts, latest evals
    python app/generate_visualizations.py --chart mrr qtype  # a subset
    python app/generate_visualizations.py --metric recall -k 5
    python app/generate_visualizations.py --checkpoint best_config_found
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

from app.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

KS = [1, 3, 5, 10]
K_METRICS = ["recall", "precision", "ndcg"]
SCALAR_METRICS = ["mrr", "map"]

# Categorical palette (validated): color follows the retrieval backend, the
# darker step of the same hue marks the -large embedding model. The aqua and
# yellow slots sit below 3:1 on white, so bars always carry direct value labels.
C_BLUE, C_BLUE_D = "#2a78d6", "#1c5cab"
C_AQUA, C_AQUA_D = "#1baf7a", "#12855c"
C_YELLOW, C_YELLOW_D = "#eda100", "#c98500"
GRID_COLOR = "#d5d4d0"

METHOD_COLORS = {"chromadb": C_BLUE, "milvus": C_AQUA, "bm25": C_YELLOW}
INDEX_COLORS = {
    ("chromadb", "small"): C_BLUE,
    ("chromadb", "large"): C_BLUE_D,
    ("milvus", "small"): C_AQUA,
    ("milvus", "large"): C_AQUA_D,
    ("bm25", "word"): C_YELLOW,
    ("bm25", "simple"): C_YELLOW_D,
}
QTYPE_COLORS = {"direct": C_BLUE, "inference": C_AQUA, "paraphrased": C_YELLOW}
# In the recall curves dashed marks the -large model; BM25 tokenizer variants
# never share a panel, so both stay solid.
MODEL_LINESTYLE = {"small": "-", "large": "--"}

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
)
DIV_CMAP = LinearSegmentedColormap.from_list("div_blue_red", ["#2a78d6", "#f0efec", "#e34948"])


# --------------------------------------------------------------------------- loading


def _find_eval_dirs() -> list[Path]:
    """Dataset dirs holding an evaluations/ folder, newest data run first."""
    return sorted(
        (p.parent for p in DATA_DIR.glob("*/*/*/evaluations") if any(p.glob("eval_*.json"))),
        key=lambda d: d.parent.name,
        reverse=True,
    )


def _pick_dataset(explicit: str | None) -> Path:
    if explicit:
        dataset = (PROJECT_ROOT / explicit).resolve()
        if not (dataset / "evaluations").is_dir():
            sys.exit(f"No evaluations/ under {dataset}")
        return dataset
    dirs = _find_eval_dirs()
    if not dirs:
        sys.exit(f"No */*/*/evaluations dirs with eval_*.json under {DATA_DIR}. Run eval_retrieval.py first.")
    if len(dirs) > 1 and sys.stdin.isatty():
        for i, d in enumerate(dirs, 1):
            n = len(list((d / "evaluations").glob("eval_*.json")))
            print(f"  {i}. {d.relative_to(DATA_DIR)}  ({n} eval files)")
        choice = input(f"\nChoose a dataset (number, Enter for [1]): ").strip()
        if choice:
            dirs[0] = dirs[int(choice) - 1]
    return dirs[0]


def _chunk_label(chunk_run: str) -> str:
    """'.../20260713_075540_chunk_fixed_size_256_50' -> 'fixed_size 256/50'."""
    name = Path(chunk_run).name
    m = re.search(r"chunk_(?P<method>.+?)_(?P<size>\d+)_(?P<overlap>\d+)$", name)
    if m:
        return f"{m['method']} {m['size']}/{m['overlap']}"
    return re.sub(r"^\d{8}_\d{6}_chunk_", "", name)


def _index_parts(meta: dict) -> tuple[str, str, str]:
    """(method, model_short, index_label) for one eval file's metadata."""
    if meta["db_type"] == "bm25":
        tok = meta.get("tokenizer", "word")
        return "bm25", tok, f"BM25 ({tok})"
    model = meta.get("embedding_model", "?")
    short = "large" if model.endswith("large") else "small"
    return meta["db_type"], short, f"{meta['db_type']} · 3-{short}"


def load_experiments(dataset: Path, all_evals: bool) -> pd.DataFrame:
    """One row per experiment (chunk config x index), latest eval per combo
    unless all_evals. Metric columns: mrr, map, recall@K/precision@K/ndcg@K
    for every K the eval recorded (NaN where a file used fewer ks)."""
    rows = []
    for f in sorted((dataset / "evaluations").glob("eval_*.json")):
        if f.name.startswith("eval_summary"):
            continue
        data = json.loads(f.read_text())
        meta, overall = data["metadata"], data["aggregates"]["overall"]
        method, model, index_label = _index_parts(meta)
        chunk = _chunk_label(meta["chunk_run"])
        row = {
            "eval_file": f.name,
            "eval_ts": re.search(r"eval_(\d{8}_\d{6})", f.name).group(1),
            "chunk": chunk,
            "method": method,
            "model": model,
            "index": index_label,
            "experiment": f"{chunk} | {index_label}",
            "num_questions": overall["num_questions"],
            "ks": meta["ks"],
            "by_question_type": data["aggregates"]["by_question_type"],
            "mrr": overall["mrr"],
            "map": overall["map"],
        }
        for k in KS:
            for m in K_METRICS:
                row[f"{m}@{k}"] = overall.get(f"{m}@{k}")
        rows.append(row)
    if not rows:
        sys.exit(f"No eval_*.json files under {dataset / 'evaluations'}")
    df = pd.DataFrame(rows)
    if not all_evals:
        df = (
            df.sort_values("eval_ts")
            .groupby(["chunk", "index"], as_index=False)
            .last()
        )
    else:
        df["experiment"] = df["experiment"] + " @" + df["eval_ts"].str[9:]
    return df.sort_values(["chunk", "index"]).reset_index(drop=True)


def _metric_col(metric: str, k: int) -> str:
    return metric if metric in SCALAR_METRICS else f"{metric}@{k}"


def _metric_label(metric: str, k: int) -> str:
    return metric.upper() if metric in SCALAR_METRICS else f"{metric.capitalize()}@{k}"


def _require(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Drop experiments missing a metric column (eval ran with fewer ks)."""
    missing = df[df[col].isna()]
    if not missing.empty:
        log.warning(
            "  dropping %d experiment(s) without %s: %s",
            len(missing),
            col,
            ", ".join(missing["experiment"]),
        )
    return df[df[col].notna()]


# --------------------------------------------------------------------------- helpers


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _bar_labels(ax, bars, fmt: str = "{:.2f}") -> None:
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.008,
            fmt.format(b.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#52514e",
        )


def _method_legend(ax, methods: list[str]) -> None:
    handles = [Patch(color=METHOD_COLORS[m], label="BM25 (lexical)" if m == "bm25" else f"Vector ({m})") for m in methods]
    ax.legend(handles=handles, title="Retrieval method", frameon=False, fontsize=9)


def _save(fig, out_dir: Path, name: str, show: bool) -> None:
    fig.tight_layout()
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, dpi=150)
    log.info("  wrote %s", out_path.relative_to(PROJECT_ROOT))
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- charts


def plot_mrr(df: pd.DataFrame, out_dir: Path, show: bool, metric: str, k: int) -> None:
    """Required chart 1: MRR per experiment, color-coded by retrieval method."""
    d = df.sort_values("mrr", ascending=False)
    fig, ax = plt.subplots(figsize=(max(9, 0.85 * len(d)), 6))
    bars = ax.bar(
        d["experiment"],
        d["mrr"],
        color=[METHOD_COLORS[m] for m in d["method"]],
        width=0.62,
    )
    _bar_labels(ax, bars)
    ax.axhline(0.85, color="#52514e", linestyle="--", linewidth=0.8)
    ax.text(ax.get_xlim()[1], 0.85, " target\n 0.85", va="center", fontsize=8, color="#52514e")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("MRR")
    ax.set_title("MRR by Experiment Configuration\n(chunk config x index, best first)", fontsize=13)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["experiment"], rotation=30, ha="right", fontsize=8)
    _style(ax)
    _method_legend(ax, [m for m in METHOD_COLORS if m in set(d["method"])])
    _save(fig, out_dir, "mrr_comparison", show)


def plot_scatter(df: pd.DataFrame, out_dir: Path, show: bool, metric: str, k: int) -> None:
    """Required chart 2: Recall@K vs Precision@K, top-5 (by MRR) labeled.
    With one gold chunk per question precision@k == recall@k / k, so points
    fall on that line; drawn as a reference."""
    rc, pc = f"recall@{k}", f"precision@{k}"
    d = _require(df, rc)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1 / k], color=GRID_COLOR, linestyle="--", linewidth=1, zorder=1)
    ax.text(0.98, 1 / k, f"precision = recall/{k}\n(1 gold chunk cap)", ha="right", va="bottom", fontsize=8, color="#52514e")
    for method in METHOD_COLORS:
        sub = d[d["method"] == method]
        if sub.empty:
            continue
        ax.scatter(sub[rc], sub[pc], s=70, color=METHOD_COLORS[method], edgecolors="white", linewidths=1.5, zorder=3)
    # stagger label offsets left/right of the point: top experiments cluster
    # tightly on the cap line, straight offsets to one side collide
    offsets = [(12, 12), (-12, -22), (12, -40), (-12, 30), (12, 48)]
    for (_, r), off in zip(d.nlargest(5, "mrr").iterrows(), offsets):
        ax.annotate(
            r["experiment"],
            (r[rc], r[pc]),
            xytext=off,
            textcoords="offset points",
            ha="left" if off[0] > 0 else "right",
            fontsize=7.5,
            color="#0b0b0b",
            arrowprops={"arrowstyle": "-", "color": GRID_COLOR, "shrinkB": 3},
        )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1 / k * 1.15)
    ax.set_xlabel(f"Recall@{k}")
    ax.set_ylabel(f"Precision@{k}")
    ax.set_title(f"Recall@{k} vs Precision@{k} per Experiment\n(top 5 by MRR labeled)", fontsize=13)
    ax.grid(color=GRID_COLOR, linewidth=0.7)
    ax.set_axisbelow(True)
    _method_legend(ax, [m for m in METHOD_COLORS if m in set(d["method"])])
    _save(fig, out_dir, "recall_vs_precision_scatter", show)


def plot_heatmap(df: pd.DataFrame, out_dir: Path, show: bool, metric: str, k: int) -> None:
    """Required chart 3: chunk config x index heatmap of the chosen metric."""
    col = _metric_col(metric, k)
    d = _require(df, col)
    mat = d.pivot_table(index="chunk", columns="index", values=col)
    fig, ax = plt.subplots(figsize=(1.6 * len(mat.columns) + 3, 0.8 * len(mat) + 2.5))
    cmap = SEQ_CMAP.copy()
    cmap.set_bad("#f0efec")  # neutral gray: combination not evaluated (vs light blue = low)
    im = ax.imshow(mat.to_numpy(na_value=float("nan")), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.55 else "#0b0b0b")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color="#52514e")
    label = _metric_label(metric, k)
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    ax.set_title(f"{label} by Chunking Config and Index\n(embedding model / BM25 per column)", fontsize=12)
    ax.set_xlabel("Index (vector DB · embedding model, or BM25)")
    ax.set_ylabel("Chunking config (method size/overlap)")
    _save(fig, out_dir, f"chunking_strategy_heatmap_{col.replace('@', '_at_')}", show)


def plot_retrieval(df: pd.DataFrame, out_dir: Path, show: bool, metric: str, k: int) -> None:
    """Required chart 4: grouped bars comparing every index per chunk config."""
    col = _metric_col(metric, k)
    d = _require(df, col)
    chunks = sorted(d["chunk"].unique())
    present = sorted(d["index"].unique())
    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(chunks) * len(present) * width + 4), 6))
    for j, idx in enumerate(present):
        sub = d[d["index"] == idx].set_index("chunk").reindex(chunks)
        color = INDEX_COLORS.get(
            (sub["method"].dropna().iloc[0], sub["model"].dropna().iloc[0]), C_YELLOW
        )
        pos = [
            (i + (j - (len(present) - 1) / 2) * width, v)
            for i, v in enumerate(sub[col])
            if pd.notna(v)  # skip combinations this index was never evaluated on
        ]
        bars = ax.bar([x for x, _ in pos], [v for _, v in pos], width=width * 0.92, color=color, label=idx)
        _bar_labels(ax, bars)
    label = _metric_label(metric, k)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(label)
    ax.set_xlabel("Chunking config")
    ax.set_xticks(range(len(chunks)))
    ax.set_xticklabels(chunks)
    ax.set_title(f"Retrieval Method Comparison: {label} per Chunking Config", fontsize=13)
    _style(ax)
    ax.legend(title="Index", frameon=False, fontsize=9)
    _save(fig, out_dir, f"retrieval_method_comparison_{col.replace('@', '_at_')}", show)


def plot_correlation(df: pd.DataFrame, out_dir: Path, show: bool, metric: str, k: int) -> None:
    """Required chart 5: correlation of all IR metrics across experiments.
    map == mrr by construction (single gold chunk), so map is omitted."""
    cols = ["mrr"] + [f"{m}@{kk}" for m in K_METRICS for kk in KS]
    d = df[cols].dropna(axis=1, how="all")
    corr = d.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3, 0.62 * n + 2.5))
    im = ax.imshow(corr.values, cmap=DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.65 else "#0b0b0b")
    fig.colorbar(im, ax=ax, label="Pearson correlation", shrink=0.8)
    ax.set_title(
        f"IR Metric Correlation Across {len(df)} Experiments\n(MAP omitted: equals MRR with one gold chunk)",
        fontsize=12,
    )
    _save(fig, out_dir, "metric_correlation_matrix", show)


def plot_recall_curve(df: pd.DataFrame, out_dir: Path, show: bool, metric: str, k: int) -> None:
    """Extra: Recall@K as K grows, small multiples per chunk config. Hue is
    the retrieval backend, dashed marks the -large embedding model."""
    chunks = sorted(df["chunk"].unique())
    fig, axes = plt.subplots(
        1, len(chunks), figsize=(5.2 * len(chunks) + 1, 5), sharey=True, squeeze=False
    )
    for ax, chunk in zip(axes[0], chunks):
        sub = df[df["chunk"] == chunk]
        for _, r in sub.iterrows():
            ks = [kk for kk in KS if pd.notna(r[f"recall@{kk}"])]
            ax.plot(
                ks,
                [r[f"recall@{kk}"] for kk in ks],
                color=METHOD_COLORS[r["method"]],
                linestyle=MODEL_LINESTYLE.get(r["model"], "-"),
                marker="o",
                markersize=6,
                linewidth=2,
                markeredgecolor="white",
                markeredgewidth=1,
            )
        ax.set_title(chunk, fontsize=11)
        ax.set_xticks(KS)
        ax.set_xlabel("K")
        ax.set_ylim(0, 1.02)
        ax.grid(color=GRID_COLOR, linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Recall@K")
    handles = [
        Line2D([], [], color=METHOD_COLORS[m], linewidth=2,
               label="BM25 (lexical)" if m == "bm25" else f"Vector ({m})")
        for m in METHOD_COLORS
        if m in set(df["method"])
    ] + [
        Line2D([], [], color="#52514e", linestyle=ls, linewidth=2, label=f"3-{mod}")
        for mod, ls in (("small", "-"), ("large", "--"))
        if mod in set(df["model"])
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Recall@K vs K per Index, by Chunking Config", fontsize=13)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, out_dir, "recall_at_k_curves", show)


def plot_qtype(df: pd.DataFrame, out_dir: Path, show: bool, metric: str, k: int) -> None:
    """Extra: chosen metric per question type (direct/inference/paraphrased)
    for every experiment - shows which question styles retrieval struggles on."""
    key = _metric_col(metric, k)  # aggregate keys use the same names
    d = df.sort_values("mrr", ascending=False)
    qtypes = list(QTYPE_COLORS)
    rows, dropped = [], []
    for _, r in d.iterrows():
        per_type = {qt: r["by_question_type"].get(qt, {}).get(key) for qt in qtypes}
        if any(v is None for v in per_type.values()):
            dropped.append(r["experiment"])
            continue
        rows.append({"experiment": r["experiment"], **per_type})
    if dropped:
        log.warning("  dropping %d experiment(s) without %s by question type: %s", len(dropped), key, ", ".join(dropped))
    data = pd.DataFrame(rows)
    if data.empty:
        log.warning("  skipping qtype chart: no experiments carry %s per question type", key)
        return
    width = 0.8 / len(qtypes)
    fig, ax = plt.subplots(figsize=(max(9, 0.95 * len(data)), 6))
    for j, qt in enumerate(qtypes):
        xs = [i + (j - (len(qtypes) - 1) / 2) * width for i in range(len(data))]
        bars = ax.bar(xs, data[qt], width=width * 0.92, color=QTYPE_COLORS[qt], label=qt)
        _bar_labels(ax, bars)
    label = _metric_label(metric, k)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(label)
    ax.set_title(f"{label} by Question Type per Experiment\n(experiments ordered by overall MRR)", fontsize=13)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data["experiment"], rotation=30, ha="right", fontsize=8)
    _style(ax)
    ax.legend(title="Question type", frameon=False, fontsize=9)
    _save(fig, out_dir, f"question_type_{key.replace('@', '_at_')}", show)


# --------------------------------------------------------------------------- driver

CHARTS = {
    "mrr": plot_mrr,
    "scatter": plot_scatter,
    "heatmap": plot_heatmap,
    "retrieval": plot_retrieval,
    "correlation": plot_correlation,
    "recall-curve": plot_recall_curve,
    "qtype": plot_qtype,
}


def _dir_suffix(i: int) -> str:
    """Spreadsheet-style suffix: 1->'a', 26->'z', 27->'aa'."""
    s = ""
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(ord("a") + r) + s
    return s


def _fresh_checkpoint_dir(base: Path) -> tuple[Path, bool]:
    """base if free, else base+'a'/'b'/... . Never returns an existing dir."""
    if not base.exists():
        return base, False
    i = 1
    while (candidate := base.with_name(base.name + _dir_suffix(i))).exists():
        i += 1
    return candidate, True


def _warn_checkpoint_renamed(requested: Path, used: Path) -> None:
    bar = "!" * 74
    log.warning("\n%s", bar)
    for ln in (
        "CHECKPOINT DIRECTORY ALREADY EXISTED — NOTHING WAS OVERWRITTEN",
        f"requested: visualizations/{requested.name}/",
        f"saved to:  visualizations/{used.name}/",
    ):
        log.warning("!!%s!!", ln.center(70))
    log.warning("%s\n", bar)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--chart",
        nargs="+",
        choices=["all", *CHARTS],
        default=["all"],
        help="Chart(s) to generate (default: all).",
    )
    parser.add_argument(
        "--metric",
        choices=SCALAR_METRICS + K_METRICS,
        default="mrr",
        help="Metric for the heatmap/retrieval/qtype charts (default: mrr).",
    )
    parser.add_argument(
        "-k",
        type=int,
        choices=KS,
        default=5,
        help="Cutoff K for @K metrics and the scatter plot (default: 5).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Dataset dir holding evaluations/ (default: menu / newest data run).",
    )
    parser.add_argument(
        "--all-evals",
        action="store_true",
        help="Keep every eval file instead of only the latest per chunk config x index.",
    )
    parser.add_argument("--no-show", action="store_true", help="Save PNGs without opening windows.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="NAME",
        help="Save into visualizations/NAME/ to capture a milestone "
        "(never overwrites an existing checkpoint; appends a/b/... instead).",
    )
    args = parser.parse_args()

    log_file = setup_logging("generate_visualizations")
    dataset = _pick_dataset(args.data)
    df = load_experiments(dataset, args.all_evals)

    renamed_from: Path | None = None
    if args.checkpoint:
        requested = PROJECT_ROOT / "visualizations" / args.checkpoint
        out_dir, renamed = _fresh_checkpoint_dir(requested)
        if renamed:
            renamed_from = requested
        out_dir.mkdir(parents=True)
    else:
        out_dir = PROJECT_ROOT / "visualizations" / datetime.now().strftime("%Y%m%d_%H%M")
        out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Dataset: %s", dataset.relative_to(PROJECT_ROOT))
    log.info(
        "Experiments: %d (%d chunk configs x %d indexes)%s",
        len(df),
        df["chunk"].nunique(),
        df["index"].nunique(),
        "" if args.all_evals else ", latest eval per combo",
    )
    log.info("Metric: %s, k=%d", args.metric, args.k)
    log.info("Output: %s", out_dir.relative_to(PROJECT_ROOT))
    if renamed_from is not None:
        _warn_checkpoint_renamed(renamed_from, out_dir)

    wanted = list(CHARTS) if "all" in args.chart else args.chart
    show = not args.no_show and args.checkpoint is None  # checkpoints: save only
    failed: list[str] = []
    for name in wanted:
        try:
            CHARTS[name](df, out_dir, show, args.metric, args.k)
        except Exception:  # noqa: BLE001 - one bad chart shouldn't kill the rest
            log.exception("ERROR building chart %r; skipping", name)
            failed.append(name)

    log.info(
        "Done: %d/%d chart(s) built%s",
        len(wanted) - len(failed),
        len(wanted),
        f"; failed: {', '.join(failed)}" if failed else "",
    )
    log.info("Log -> %s", log_file)
    if renamed_from is not None:  # re-log so it isn't scrolled off by the "wrote ..." lines
        _warn_checkpoint_renamed(renamed_from, out_dir)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
