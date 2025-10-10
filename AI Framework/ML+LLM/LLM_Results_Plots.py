import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from textwrap import shorten

# -----------------------------
# Paths & params
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / "LLM Agent" / "Outputs"   # <- updated input folder
if not INPUT_DIR.is_dir():
    raise FileNotFoundError(f"Expected CSVs under: {INPUT_DIR}")

llm_files = sorted(INPUT_DIR.glob("*.csv"))
if not llm_files:
    raise FileNotFoundError(f"No CSVs found in: {INPUT_DIR}")

OUT_DIR = BASE_DIR / "Results" / "LLM_Summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 15
MAX_LABEL_CHARS = 38

# -----------------------------
# Trait metadata (ID -> label)
# -----------------------------
TRAIT_META = {
    "T01": "Talks Excessively",
    "T02": "Rapid Speech (Pressured)",
    "T03": "Interruptive Speech",
    "T04": "Topic Switching / Tangential",
    "T05": "Blurting Out Answers",
    "T06": "Difficulty Waiting Turn",
    "T07": "Fidgeting & Vocal Restlessness",
    "T08": "Disfluency / Stuttering-like",
    "T09": "Echolalia / Repetition",
    "T10": "Stilted / Pedantic Speech",
    "T11": "Emotional Reactivity in Speech",
    "T12": "Inattentiveness",
    "T13": "Impulsive Responses",
    "T14": "Working Memory Lapses",
    "T15": "Motor Restlessness",
    "T16": "Delay Aversion",
    "T17": "Overactive Gesturing",
    "T18": "Sensory Distractibility",
    "T19": "Reward-Seeking Urgency",
    "T20": "Emotional Dysregulation",
    "T21": "Perseverative Speech",
    "T22": "Monosyllabic Responses",
    "T23": "Excessive Inquiries",
    "T24": "Rapid Topic Overrun",
    "T25": "Sentence Fragments",
    "T26": "Vocal Fillers Overuse",
    "T27": "Rapid Laughter",
    "T28": "Abrupt Volume Changes",
    "T29": "Garbled Articulation",
    "T30": "Frequent Self-Correction",
    "T31": "Auditory Overload Response",
    "T32": "Sensory Hyperfocus",
    "T33": "Nonstop Oral Motor Noise",
    "T34": "Peripheral Awareness Distraction",
    "T35": "Rapid Topic Recap",
    "T36": "Over-enthusiastic Tone",
    "T37": "Forced Humor",
    "T38": "Distracted Eye Contact",
    "T39": "Repetitive Questioning",
    "T40": "Meta-Talk Overuse",
    "T41": "Peripheral Vocalizations",
    "T42": "Segmented Speech",
    "T43": "Excessive Qualifiers",
    "T44": "Pseudo-Questions",
    "T45": "Immediate Repair",
    "T46": "Spatial Drift",
    "T47": "Content Overload",
    "T48": "Rapid Ideation",
    "T49": "Externalization",
    "T50": "Anticipatory Speech",
}

# -----------------------------
# Utilities
# -----------------------------
def pick_col(df: pd.DataFrame, *cands: str):
    m = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in m:
            return m[c.lower()]
    return None

LABELED = re.compile(r"\bT(\d{2})\s*\(([^)]+)\)", re.I)
BARE    = re.compile(r"\bT(\d{2})\b", re.I)

def parse_traits(cell) -> list[str]:
    """Extract Txx codes if present; keep order and de-duplicate."""
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.upper() == "NONE":
        return []
    out = []
    for m in LABELED.finditer(s):
        out.append(f"T{m.group(1)}")
    if not out:
        for m in BARE.finditer(s):
            out.append(f"T{m.group(1)}")
    if not out:
        for tok in re.split(r"[;,|]", s):
            t = tok.strip()
            if t and t.upper() != "NONE":
                out.append(t)
    seen, dedup = set(), []
    for t in out:
        if t not in seen:
            dedup.append(t)
            seen.add(t)
    return dedup

def trait_id_from_token(tok: str) -> str | None:
    m = BARE.search(tok)
    return f"T{m.group(1)}" if m else None

def trait_label_from_id(tid: str) -> str:
    return TRAIT_META.get(tid, tid)

def display_name(tid: str) -> str:
    return shorten(f"{tid} – {trait_label_from_id(tid)}", width=MAX_LABEL_CHARS, placeholder="…")

def load_one(path: Path) -> pd.DataFrame:
    """Load one CSV and normalize columns + scores + traits."""
    df = pd.read_csv(path)
    col_tr  = pick_col(df, "traits", "trait")
    col_spk = pick_col(df, "speaker")
    col_utt = pick_col(df, "utterance", "text", "content")
    col_c   = pick_col(df, "clarityscore", "clarity", "clarity_score")
    col_u   = pick_col(df, "uniquenessscore", "uniqueness", "uniqueness_score")
    col_q   = pick_col(df, "qualityscore", "quality", "quality_score")

    out = pd.DataFrame({
        "child_file": path.stem,  # this CSV = this child
        "speaker":    df[col_spk] if col_spk else np.nan,
        "utterance":  df[col_utt] if col_utt else "",
        "traits_raw": df[col_tr]  if col_tr  else "",
        "clarity":    pd.to_numeric(df[col_c], errors="coerce") if col_c else np.nan,
        "uniqueness": pd.to_numeric(df[col_u], errors="coerce") if col_u else np.nan,
        "quality":    pd.to_numeric(df[col_q], errors="coerce") if col_q else np.nan,
    })
    out["overall_confidence"] = out[["clarity","uniqueness","quality"]].mean(axis=1, skipna=True)
    out["traits_list"] = out["traits_raw"].apply(parse_traits)
    out["is_flagged"]  = out["traits_list"].apply(lambda lst: len(lst) > 0)
    return out

# -----------------------------
# Load all data
# -----------------------------
frames = []
for f in llm_files:
    try:
        frames.append(load_one(f))
    except Exception as e:
        print(f"skipping {f.name}: {e}")
if not frames:
    raise RuntimeError("No valid CSV after parsing.")
data = pd.concat(frames, ignore_index=True)

# Flagged utterances only
flagged = data[data["is_flagged"]].copy()

# Expand traits: 1 row per trait hit
exp = (
    flagged.assign(_trait=flagged["traits_list"])
           .explode("_trait", ignore_index=True)
           .dropna(subset=["_trait"])
)
exp["trait_id"]    = exp["_trait"].apply(lambda s: trait_id_from_token(str(s)))
exp = exp.dropna(subset=["trait_id"]).copy()
exp["trait_id"]    = exp["trait_id"].astype(str)
exp["trait_label"] = exp["trait_id"].apply(trait_label_from_id)
exp["trait_name"]  = exp["trait_id"].apply(display_name)

# -----------------------------
# Global summaries
# -----------------------------
trait_counts = (
    exp.groupby(["trait_id","trait_label","trait_name"], as_index=False)
       .size().rename(columns={"size":"count"})
       .sort_values("count", ascending=False)
)
trait_counts.to_csv(OUT_DIR / "trait_counts_global.csv", index=False)
top_traits = trait_counts.head(TOP_N).copy()

trait_stats = (
    exp.groupby(["trait_id","trait_label","trait_name"], as_index=False)
       .agg(n=("trait_id","size"),
            mean_overall=("overall_confidence","mean"),
            sd_overall=("overall_confidence","std"))
)
trait_stats["se_overall"] = trait_stats["sd_overall"] / np.sqrt(trait_stats["n"].clip(lower=1))
trait_stats = trait_stats.merge(
    top_traits[["trait_id"]].assign(rank=np.arange(len(top_traits))),
    on="trait_id", how="inner"
).sort_values("rank")
trait_stats.to_csv(OUT_DIR / "trait_confidence_topN.csv", index=False)

global_scores = (flagged[["clarity","uniqueness","quality","overall_confidence"]]
                 .mean().to_frame("mean").reset_index()
                 .rename(columns={"index":"metric"}))
global_scores.to_csv(OUT_DIR / "global_scores_flagged.csv", index=False)

# -----------------------------
# ADHD: full T01–T50 with severity weights → per-child severity index
# -----------------------------
ALL_ADHD_TRAITS = {f"T{i:02d}" for i in range(1, 51)}
ADHD_WEIGHTS = {
    "T01":3,"T02":3,"T03":3,"T04":2,"T05":2,"T06":2,"T07":2,"T08":2,"T09":1,"T10":1,
    "T11":2,"T12":2,"T13":2,"T14":1,"T15":2,"T16":2,"T17":1,"T18":1,"T19":1,"T20":2,
    "T21":2,"T22":1,"T23":2,"T24":2,"T25":1,"T26":2,"T27":1,"T28":2,"T29":1,"T30":1,
    "T31":2,"T32":1,"T33":1,"T34":1,"T35":1,"T36":1,"T37":1,"T38":1,"T39":1,"T40":1,
    "T41":1,"T42":1,"T43":1,"T44":1,"T45":1,"T46":1,"T47":2,"T48":2,"T49":1,"T50":1,
}
MAX_W = 3  # max severity weight in the scale

exp["is_adhd_trait"] = exp["trait_id"].isin(ALL_ADHD_TRAITS)
exp["adhd_weight"] = exp.apply(
    lambda r: ADHD_WEIGHTS.get(r["trait_id"], 1) if r["is_adhd_trait"] else 0, axis=1
)

per_child = (
    exp.groupby("child_file", as_index=False)
       .agg(
           total_traits=("trait_id", "count"),
           adhd_count=("is_adhd_trait", "sum"),
           adhd_weight_sum=("adhd_weight", "sum"),
       )
)

# Unweighted percent (for reference)
per_child["adhd_percent_unweighted"] = np.where(
    per_child["total_traits"] > 0,
    per_child["adhd_count"] / per_child["total_traits"] * 100.0,
    np.nan
)

# Severity-weighted index (0–100)
per_child["adhd_index_weighted_pct"] = np.where(
    per_child["total_traits"] > 0,
    (per_child["adhd_weight_sum"] / (MAX_W * per_child["total_traits"])) * 100.0,
    np.nan
)

per_child[[
    "child_file","total_traits","adhd_count","adhd_weight_sum",
    "adhd_percent_unweighted","adhd_index_weighted_pct"
]].to_csv(OUT_DIR / "adhd_percent_per_child.csv", index=False)

# -----------------------------
# PLOTS (key ones) — tight layout
# -----------------------------

# Top-N trait counts (horizontal)
fig, ax = plt.subplots(figsize=(10, 5))
sub = top_traits.copy()
ax.barh(sub["trait_name"][::-1], sub["count"][::-1])
ax.set_xlabel("Count (across all files)")
ax.set_ylabel("Trait")
ax.set_title(f"Top {TOP_N} traits detected by LLM")
ax.margins(y=0)
plt.tight_layout()
plt.show()

# Mean confidence for Top-N (vertical)
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(trait_stats))
y = trait_stats["mean_overall"].values
se = trait_stats["se_overall"].values
labs = trait_stats["trait_id"].tolist()
ax.bar(x, y, yerr=se, capsize=4)
ax.set_xticks(x, labs, rotation=45, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Mean confidence (0–1)")
ax.set_title(f"Top {TOP_N} traits: mean confidence")
ax.margins(x=0)
for xi, v, n in zip(x, y, trait_stats["n"].values):
    ax.text(xi, min(v + 0.03, 0.98), f"n={int(n)}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()

# Global scores (flagged-only)
labels = [lbl.replace("_"," ").capitalize() for lbl in global_scores["metric"]]
vals   = global_scores["mean"].astype(float).values
ymax   = min(1.05, max(0.1, vals.max()) * 1.15)
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, vals)
ax.set_ylim(0, ymax)
ax.set_ylabel("Mean on flagged utterances (0–1)")
ax.set_title("Global mean scores (flagged)")
ax.margins(x=0)
for rect, v in zip(bars, vals):
    y_text = min(v + ymax*0.03, ymax - ymax*0.02)
    ax.text(rect.get_x() + rect.get_width()/2, y_text, f"{v:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# -----------------------------
# ADHD per child (row-level): % rows with ≥1 ADHD trait (T01–T50)
# -----------------------------

ADHD_TRAITS = {f"T{i:02d}" for i in range(1, 51)}

def row_has_adhd(traits_list):
    """True if this row has at least one ADHD-like Txx."""
    if not isinstance(traits_list, (list, tuple)) or len(traits_list) == 0:
        return False
    for tok in traits_list:
        tid = trait_id_from_token(str(tok))
        if tid in ADHD_TRAITS:
            return True
    return False

# Row-level flag on the full dataset (all rows)
data["traits_list"] = data.get("traits_list", data.get("traits_raw", "")).apply(parse_traits) if "traits_list" not in data else data["traits_list"]
data["has_adhd_row"] = data["traits_list"].apply(row_has_adhd)

# Aggregate per child/file: count rows, count ADHD-rows, compute %
per_child_rows = (
    data.groupby("child_file", as_index=False)
        .agg(
            total_rows=("traits_list", "size"),
            adhd_rows=("has_adhd_row", "sum"),
        )
)
per_child_rows["adhd_row_percent"] = np.where(
    per_child_rows["total_rows"] > 0,
    per_child_rows["adhd_rows"] / per_child_rows["total_rows"] * 100.0,
    np.nan
)

# Save the table for reporting
per_child_rows.to_csv(OUT_DIR / "adhd_row_percent_per_child.csv", index=False)

# -----------------------------
# Plot: ADHD% per child (CSV names as labels)
# -----------------------------
if not per_child_rows["adhd_row_percent"].dropna().empty:
    plot_df = per_child_rows.sort_values("child_file")   # alphabetical

    height_in = max(6, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, height_in))

    y_pos = np.arange(len(plot_df))
    vals  = plot_df["adhd_row_percent"].values
    labs  = plot_df["child_file"].tolist()

    bars = ax.barh(y_pos, vals)
    ax.set_yticks(y_pos, labs)

    ax.set_xlabel("% of rows with ≥1 ADHD-like trait")
    ax.set_ylabel("Child (CSV file)")
    ax.set_title("ADHD-like presence per child (row-level %)")

    ax.margins(y=0)

    xmax = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 0.0
    ax.set_xlim(0, xmax * 1.05 if xmax > 0 else 1.0)

    for rect, v in zip(bars, vals):
        if np.isnan(v):
            continue
        ax.annotate(f"{v:.1f}%",
                    xy=(v, rect.get_y() + rect.get_height()/2),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9, clip_on=True)

    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()
else:
    print("No ADHD row-level percentages to plot.")

