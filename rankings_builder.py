import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────────
# CLI ARGUMENTS
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Book Club Rankings Processor")
parser.add_argument("--rankings", required=True, help="Path to rankings CSV/TSV")
parser.add_argument("--submissions", required=True, help="Path to submissions CSV/TSV")
args = parser.parse_args()


# ─────────────────────────────────────────────
# LOAD INPUTS
# ─────────────────────────────────────────────
def load_file(path):
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


rankings_raw = load_file(args.rankings)
submissions_raw = load_file(args.submissions)


# ─────────────────────────────────────────────
# IDENTIFY KEY COLUMNS FLEXIBLY
# ─────────────────────────────────────────────
def find_col(df, *keywords):
    """Return first column whose name contains ALL keywords (case-insensitive)."""
    for col in df.columns:
        col_lower = col.lower()
        if all(kw.lower() in col_lower for kw in keywords):
            return col
    raise KeyError(
        f"No column matching {keywords} found. Columns: {df.columns.tolist()}"
    )


rankings_name_col = find_col(rankings_raw, "name")
rankings_time_col = find_col(rankings_raw, "timestamp")
submissions_name_col = find_col(submissions_raw, "name")
submissions_title_col = find_col(submissions_raw, "title")
submissions_author_col = find_col(submissions_raw, "author")

# ─────────────────────────────────────────────
# NORMALIZE CAPITALIZATION HELPERS
# ─────────────────────────────────────────────
# Words that should stay lowercase in title case (unless first word)
_LOWERCASE_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "but",
    "or",
    "nor",
    "for",
    "so",
    "yet",
    "at",
    "by",
    "in",
    "of",
    "on",
    "to",
    "up",
    "as",
    "it",
    "is",
}


def smart_title(text):
    """
    Title-case a string, keeping articles/conjunctions/prepositions lowercase
    unless they are the first or last word.
    """
    if not isinstance(text, str) or text.strip() == "":
        return text
    words = text.strip().split()
    result = []
    for i, word in enumerate(words):
        # Preserve all-caps abbreviations (e.g. AI, NYT)
        if word.isupper() and len(word) > 1:
            result.append(word)
        elif i == 0 or i == len(words) - 1:
            result.append(word.capitalize())
        elif word.lower() in _LOWERCASE_WORDS:
            result.append(word.lower())
        else:
            result.append(word.capitalize())
    return " ".join(result)


def normalize_name(text):
    """Capitalize first letter of each part of a person's name."""
    if not isinstance(text, str):
        return text
    return " ".join(p.capitalize() for p in text.strip().split())


# ─────────────────────────────────────────────
# PARSE RANKINGS
# ─────────────────────────────────────────────
meta_cols = {rankings_time_col, rankings_name_col}
score_cols = [c for c in rankings_raw.columns if c not in meta_cols]
scores_df = rankings_raw[score_cols].apply(pd.to_numeric, errors="coerce")

# Shorten display names: strip " - Author Name" suffix, then normalize
short_names = {col: smart_title(col.split(" - ")[0].strip()) for col in score_cols}
scores_df = scores_df.rename(columns=short_names)

# Voter first names — normalized
voter_names = (
    rankings_raw[rankings_name_col]
    .str.strip()
    .str.split()
    .str[0]
    .apply(normalize_name)
    .tolist()
)


# ─────────────────────────────────────────────
# BUILD SUBMITTER + AUTHOR MAPS FROM SUBMISSIONS
# ─────────────────────────────────────────────
def clean_submissions(df, title_col, name_col, author_col):
    """
    Return a dict: normalized_short_title -> (submitter_first, author_last)
    Skips junk rows (e.g. 'remove').
    """
    mapping = {}
    for _, row in df.iterrows():
        raw_title = str(row[title_col]).strip()
        raw_name = str(row[name_col]).strip()
        raw_author = str(row[author_col]).strip()

        if raw_title.lower() in ("remove", "nan", "") or raw_name.lower() in (
            "remove",
            "nan",
            "",
        ):
            continue

        # Short title = everything before " - " or " by " (case-insensitive)
        short = raw_title.split(" - ")[0].split(" by ")[0].strip()
        short_norm = smart_title(short)

        submitter_first = normalize_name(raw_name.split()[0])

        # Author last name: last token of author field
        author_parts = raw_author.split()
        author_last = normalize_name(author_parts[-1]) if author_parts else "Unknown"

        mapping[short_norm] = (submitter_first, author_last)
    return mapping


submission_map = clean_submissions(
    submissions_raw,
    submissions_title_col,
    submissions_name_col,
    submissions_author_col,
)


def lookup_submission(short_title):
    """
    Direct lookup first; fall back to 12-char prefix fuzzy match.
    Returns (submitter_first, author_last).
    """
    if short_title in submission_map:
        return submission_map[short_title]
    short_lower = short_title.lower()
    for key, val in submission_map.items():
        key_lower = key.lower()
        if short_lower[:12] in key_lower or key_lower[:12] in short_lower:
            return val
    return ("Unknown", "Unknown")


# ─────────────────────────────────────────────
# COMPUTE STATISTICS
# ─────────────────────────────────────────────
def compute_stats(df):
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append(
            {
                "Book": col,
                "Mean": round(s.mean(), 3),
                "Median": round(s.median(), 3),
                "Std": round(s.std(), 3),
                "IQR": round(q3 - q1, 3),
                "Range": int(s.max() - s.min()),
                "Min": int(s.min()),
                "Max": int(s.max()),
                "N": int(s.count()),
            }
        )
    return pd.DataFrame(rows)


stats_df = compute_stats(scores_df)
stats_df = stats_df.sort_values(["Mean", "IQR"], ascending=[False, True]).reset_index(
    drop=True
)
stats_df.insert(0, "Rank", stats_df.index + 1)

# ─────────────────────────────────────────────
# SUBMITTER / AUTHOR TABLE
# ─────────────────────────────────────────────
meta_rows = []
for _, row in stats_df.iterrows():
    submitter, author_last = lookup_submission(row["Book"])
    meta_rows.append(
        {
            "Rank": row["Rank"],
            "Book": row["Book"],
            "Submitter": submitter,
            "Author": author_last,
        }
    )
meta_df = pd.DataFrame(meta_rows)


# ─────────────────────────────────────────────
# PRINT MARKDOWN TABLES
# ─────────────────────────────────────────────
def df_to_markdown(df):
    col_widths = {
        c: max(len(str(c)), df[c].astype(str).map(len).max()) for c in df.columns
    }
    header = "| " + " | ".join(str(c).ljust(col_widths[c]) for c in df.columns) + " |"
    sep = "| " + " | ".join("-" * col_widths[c] for c in df.columns) + " |"
    rows = [
        "| "
        + " | ".join(str(v).ljust(col_widths[c]) for c, v in zip(df.columns, r))
        + " |"
        for r in df.itertuples(index=False)
    ]
    return "\n".join([header, sep] + rows)


# ─────────────────────────────────────────────
# OUTPUT 1: ALL METRICS → CSV
# ─────────────────────────────────────────────
metrics_output = "book_rankings_metrics.csv"
stats_df.to_csv(metrics_output, index=False)
print(f"Metrics saved: {metrics_output}")

# ─────────────────────────────────────────────
# OUTPUT 2: SUBMITTER & AUTHOR → MARKDOWN
# ─────────────────────────────────────────────
print("\n## Book Rankings — Submitter & Author\n")
print(df_to_markdown(meta_df))

# ─────────────────────────────────────────────
# FIGURE 1: KENDALL'S TAU CORRELATION HEATMAP (full mirror)
# ─────────────────────────────────────────────
n_voters = len(scores_df)
tau_matrix = np.zeros((n_voters, n_voters))

for i in range(n_voters):
    for j in range(n_voters):
        x = scores_df.iloc[i].values.astype(float)
        y = scores_df.iloc[j].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        tau_matrix[i, j] = (
            stats.kendalltau(x[mask], y[mask])[0] if mask.sum() > 1 else np.nan
        )

tau_df = pd.DataFrame(tau_matrix, index=voter_names, columns=voter_names)

fig1, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(
    tau_df,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor="white",
    ax=ax1,
    annot_kws={"size": 9},
)
ax1.set_title(
    "Kendall's Tau Correlation Matrix\n(Pairwise Voter Agreement on Book Rankings)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax1.set_xlabel("Voter", fontsize=11)
ax1.set_ylabel("Voter", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("kendall_tau_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure 1 saved: kendall_tau_heatmap.png")

# ─────────────────────────────────────────────
# FIGURE 2: RATING DISTRIBUTION BY INDIVIDUAL
# ─────────────────────────────────────────────
ncols = 3
nrows = int(np.ceil(n_voters / ncols))
palette = sns.color_palette("tab10", n_voters)

fig2, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
axes_flat = axes.flatten()

for idx, (voter, color) in enumerate(zip(voter_names, palette)):
    ax = axes_flat[idx]
    voter_scores = scores_df.iloc[idx].dropna().astype(int)
    counts = voter_scores.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)

    ax.bar(
        counts.index,
        counts.values,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        width=0.6,
    )
    mean_val = voter_scores.mean()
    ax.axvline(
        mean_val,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean={mean_val:.2f}",
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(voter, fontsize=11, fontweight="bold")
    ax.set_xlabel("Rating (1–5)", fontsize=9)
    ax.set_ylabel("# Books", fontsize=9)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlim(0.4, 5.6)
    sns.despine(ax=ax)

for idx in range(n_voters, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig2.suptitle(
    "Rating Distribution by Individual Voter", fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("rating_distribution_by_voter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure 2 saved: rating_distribution_by_voter.png")

# ─────────────────────────────────────────────
# FIGURE 3: MEAN RATING WITH IQR ERROR BARS
# ─────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(14, 7))
norm_vals = (stats_df["Mean"] - stats_df["Mean"].min()) / (
    stats_df["Mean"].max() - stats_df["Mean"].min()
)
colors = plt.cm.RdYlGn(norm_vals)

ax3.barh(
    stats_df["Book"][::-1],
    stats_df["Mean"][::-1],
    xerr=stats_df["IQR"][::-1] / 2,
    color=colors[::-1],
    edgecolor="grey",
    linewidth=0.5,
    error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "dimgrey"},
    height=0.65,
)
ax3.set_xlabel("Mean Rating (error bars = ½ IQR)", fontsize=11)
ax3.set_title(
    "Book Rankings by Mean Rating\n(color = score intensity, error bars = spread)",
    fontsize=13,
    fontweight="bold",
)
ax3.set_xlim(0, 6)
ax3.axvline(3, color="grey", linestyle=":", linewidth=1)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
sns.despine(ax=ax3)
plt.tight_layout()
plt.savefig("mean_rating_ranked.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure 3 saved: mean_rating_ranked.png")
