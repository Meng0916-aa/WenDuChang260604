"""
14_build_section_ml_dataset.py

Merge LOCAL section thermal-field features (script 13) with cross-section
quality labels into one ML-ready table, keyed by sample_id.

Input:
  - results/tables/local_section_features.csv          (from script 13)
  - section quality labels (LOCAL, git-ignored), one of:
      data/metadata/section_quality_labels.csv   (preferred)
      data/metadata/section_quality_labels.xlsx  (needs openpyxl)

Derived labels (computed here when not supplied; dilution_rate is a PERCENT):
    dilution_rate     = D_mm / (D_mm + H_mm) * 100
    aspect_ratio      = W_mm / H_mm
    wetting_angle_avg = (theta_left_deg + theta_right_deg) / 2
    wetting_angle_diff= abs(theta_left_deg - theta_right_deg)

quality_label (auto when missing) = "Good" iff ALL of:
    dilution_rate     in [rule.dilution_rate_min, rule.dilution_rate_max]
    aspect_ratio      in [rule.aspect_ratio_min, rule.aspect_ratio_max]
    wetting_angle_avg in [rule.wetting_angle_min, rule.wetting_angle_max]
    defect_presence == 0
otherwise "Bad".

Output:
  - results/tables/section_ml_dataset.csv

Safety: missing label file -> clear guidance, no fabrication. sample_ids in
features but not in labels -> ERROR (listed). sample_ids in labels but not in
features -> WARNING (listed).

Usage:
    python scripts/14_build_section_ml_dataset.py --config configs/default.yaml
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config

# Label id/measurement columns that may overlap with the feature table; on a
# sample_id merge we keep the feature-table copy of the id columns.
_OVERLAP_ID_COLS = ["experiment_id", "track_id", "section_id",
                    "section_position_mm"]

_DEFAULT_RULE = {
    "dilution_rate_min": 30.0, "dilution_rate_max": 50.0,
    "aspect_ratio_min": 3.0, "aspect_ratio_max": 6.0,
    "wetting_angle_min": 30.0, "wetting_angle_max": 55.0,
}

# Cross-section measurements that MUST be present and numeric to build a real
# label. Empty / NaN / non-numeric values are NEVER auto-labelled "Bad" — the
# build stops and reports the offending sample_ids (no label pollution).
_REQUIRED_LABEL_COLS = ["H_mm", "W_mm", "D_mm", "theta_left_deg",
                        "theta_right_deg", "defect_presence"]


class SectionLabelError(ValueError):
    """Required cross-section measurement values are missing or non-numeric."""


def _missing(series_or_none, index):
    """Boolean mask: True where the column is absent or NaN/blank."""
    if series_or_none is None:
        return pd.Series([True] * len(index), index=index)
    s = series_or_none
    return s.isna() | (s.astype(str).str.strip() == "")


def _is_invalid_value(val) -> bool:
    """True if a measurement cell is empty / NaN / not convertible to a number."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    try:
        float(val)
    except (TypeError, ValueError):
        return True
    return False


def find_missing_label_values(labels: pd.DataFrame, required=None) -> dict:
    """Locate missing/invalid required measurements.

    Returns an ordered dict {sample_id: [missing_field, ...]} for every row that
    lacks a usable value in one or more required columns. Fields are listed in
    _REQUIRED_LABEL_COLS order. An empty dict means all required values are
    present and numeric.
    """
    required = required or _REQUIRED_LABEL_COLS
    problems = {}
    for _, row in labels.iterrows():
        sid = str(row.get("sample_id", "<no sample_id>"))
        miss = []
        for col in required:
            if col not in labels.columns or _is_invalid_value(row.get(col)):
                miss.append(col)
        if miss:
            problems[sid] = miss
    return problems



def compute_derived_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add dilution_rate / aspect_ratio / wetting_angle_avg / wetting_angle_diff.

    Computes a derived column for rows where it is missing/blank and the source
    measurements are available; existing user-supplied values are preserved.
    """
    out = df.copy()

    def fill(col, source_cols, formula):
        if not set(source_cols) <= set(out.columns):
            return
        need = _missing(out[col] if col in out.columns else None, out.index)
        if not need.any():
            return
        computed = formula()
        if col in out.columns:
            out[col] = out[col].where(~need, computed)
        else:
            out[col] = computed

    D = out["D_mm"] if "D_mm" in out.columns else None
    H = out["H_mm"] if "H_mm" in out.columns else None
    W = out["W_mm"] if "W_mm" in out.columns else None
    L = out["theta_left_deg"] if "theta_left_deg" in out.columns else None
    R = out["theta_right_deg"] if "theta_right_deg" in out.columns else None

    fill("dilution_rate", ["D_mm", "H_mm"], lambda: D / (D + H) * 100.0)
    fill("aspect_ratio", ["W_mm", "H_mm"], lambda: W / H)
    fill("wetting_angle_avg", ["theta_left_deg", "theta_right_deg"],
         lambda: (L + R) / 2.0)
    fill("wetting_angle_diff", ["theta_left_deg", "theta_right_deg"],
         lambda: (L - R).abs())
    return out


def apply_quality_rule(df: pd.DataFrame, rule: dict) -> pd.Series:
    """Compute Good/Bad per the percent-based section rule (+ defect gate).

    NOTE: this is the pure rule given numeric inputs. It does NOT decide whether
    a row is allowed to be auto-labelled — that gate lives in
    ensure_quality_label, which refuses to label rows with invalid inputs.
    """
    def num(col, default=np.nan):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series([default] * len(df), index=df.index)

    dr = num("dilution_rate")
    ar = num("aspect_ratio")
    wa = num("wetting_angle_avg")
    defect = num("defect_presence", default=0).fillna(0)

    good = (
        dr.between(rule["dilution_rate_min"], rule["dilution_rate_max"]) &
        ar.between(rule["aspect_ratio_min"], rule["aspect_ratio_max"]) &
        wa.between(rule["wetting_angle_min"], rule["wetting_angle_max"]) &
        (defect == 0)
    )
    return good.map(lambda b: "Good" if bool(b) else "Bad")


def ensure_quality_label(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Fill quality_label only for rows that are BOTH missing a label AND have
    valid rule inputs (dilution_rate, aspect_ratio, wetting_angle_avg,
    defect_presence). User-supplied labels are always preserved. Rows with
    invalid inputs are never auto-labelled — they are left blank, never "Bad".
    """
    out = df.copy()
    auto = apply_quality_rule(out, rule)

    def num(col):
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce")
        return pd.Series([np.nan] * len(out), index=out.index)

    valid = (num("dilution_rate").notna() & num("aspect_ratio").notna() &
             num("wetting_angle_avg").notna() & num("defect_presence").notna())

    need = _missing(out["quality_label"] if "quality_label" in out.columns
                    else None, out.index)
    fill = need & valid
    if "quality_label" not in out.columns:
        out["quality_label"] = pd.Series([np.nan] * len(out), index=out.index)
    # Keep existing label where NOT fill; use the auto label where fill is True.
    out["quality_label"] = out["quality_label"].where(~fill, auto)
    return out


def build_section_dataset(features: pd.DataFrame, labels: pd.DataFrame,
                          rule: dict):
    """Merge features + labels on sample_id, returning (merged, missing_feats).

    Order of checks:
      1. both tables must have a 'sample_id' column;
      2. every feature sample_id must have a label row (else ValueError);
      3. the labels that WILL be merged must carry valid required measurements
         (else SectionLabelError — no auto "Bad" pollution);
      4. derive ratios + auto quality_label, then inner-merge on sample_id.

    missing_feats lists label sample_ids that have no features (a warning, not
    fatal).
    """
    if "sample_id" not in features.columns:
        raise ValueError("local feature table has no 'sample_id' column.")
    if "sample_id" not in labels.columns:
        raise ValueError("label file has no 'sample_id' column "
                         "(see docs/section_quality_label_template.md).")

    feat_ids = set(features["sample_id"].astype(str))
    label_ids = set(labels["sample_id"].astype(str))
    missing_labels = sorted(feat_ids - label_ids)   # features without a label
    missing_feats = sorted(label_ids - feat_ids)    # labels without features

    if missing_labels:
        raise ValueError(
            f"{len(missing_labels)} section sample(s) have features but NO "
            f"label: {missing_labels}. Add their rows to the label file "
            f"(refusing to silently drop samples).")

    # Validate required measurements on the labels that will actually be merged
    # (extra, unused template rows are not validated). Empty/NaN/non-numeric
    # values stop the build instead of being auto-labelled "Bad".
    used = labels[labels["sample_id"].astype(str).isin(feat_ids)]
    problems = find_missing_label_values(used)
    if problems:
        lines = ["Missing required section label values:"]
        for sid in sorted(problems):
            lines.append(f"sample_id={sid} missing {','.join(problems[sid])}")
        raise SectionLabelError("\n".join(lines))

    # Measurements complete -> derive continuous labels and (auto) quality_label.
    labels = compute_derived_labels(labels)
    labels = ensure_quality_label(labels, rule)

    # Drop id columns from labels that already exist in features (keep feature copy).
    drop = [c for c in _OVERLAP_ID_COLS if c in labels.columns
            and c in features.columns]
    labels_for_merge = labels.drop(columns=drop)

    merged = features.merge(labels_for_merge, on="sample_id", how="inner")
    return merged, missing_feats


def _load_labels(qcfg: dict):
    csv_path = qcfg.get("label_file_csv")
    xlsx_path = qcfg.get("label_file_excel")
    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path), csv_path
    if xlsx_path and os.path.exists(xlsx_path):
        try:
            return pd.read_excel(xlsx_path), xlsx_path
        except ImportError as e:
            raise SystemExit(
                f"[14] Excel label file {xlsx_path} found but reading it needs "
                f"'openpyxl'. Install it in the conda env or provide a CSV at "
                f"{csv_path}. Underlying error: {e}")
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    scfg = config.get("section_samples", {}) or {}
    qcfg = config.get("section_quality_labels", {}) or {}
    rule = dict(_DEFAULT_RULE)
    rule.update(qcfg.get("quality_rule", {}) or {})

    feat_path = scfg.get(
        "local_feature_output",
        os.path.join(config["paths"]["results_tables"], "local_section_features.csv"))
    out_path = qcfg.get(
        "output_dataset",
        os.path.join(config["paths"]["results_tables"], "section_ml_dataset.csv"))

    if not os.path.exists(feat_path):
        print(f"[14] Local feature table not found: {feat_path}. Run script 13 first.")
        return
    features = pd.read_csv(feat_path)

    labels, label_src = _load_labels(qcfg)
    if labels is None:
        print("[14] No section quality-label file found. Expected one of:")
        print(f"        {qcfg.get('label_file_csv')}")
        print(f"        {qcfg.get('label_file_excel')}")
        print("[14] Create it from docs/section_quality_label_template.md "
              "(key column: sample_id). These files stay LOCAL.")
        return

    print(f"[14] features: {len(features)} section(s) from {feat_path}")
    print(f"[14] labels:   {len(labels)} row(s) from {label_src}")

    try:
        merged, missing_feats = build_section_dataset(features, labels, rule)
    except SectionLabelError as e:
        print(f"[14] {e}")
        print("[14] Fill data/metadata/section_quality_labels.csv with measured "
              "cross-section values before running 14.")
        sys.exit(1)
    except ValueError as e:
        raise SystemExit(f"[14] {e}")

    if missing_feats:
        print(f"[14] WARNING: {len(missing_feats)} label(s) have NO matching "
              f"features (ignored in the merge): {missing_feats}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    merged.to_csv(out_path, index=False)

    n = len(merged)
    n_exp = merged["experiment_id"].astype(str).nunique() \
        if "experiment_id" in merged.columns else 0
    label_counts = merged["quality_label"].value_counts().to_dict()

    print(f"[14] merged dataset -> {out_path} ({n} sections, "
          f"{merged.shape[1]} columns)")
    print(f"[14] unique experiments     : {n_exp}")
    print(f"[14] quality_label counts   : {label_counts}")
    print("[14] regression target summary:")
    for tgt in ("dilution_rate", "aspect_ratio", "wetting_angle_avg",
                "wetting_angle_diff"):
        if tgt in merged.columns:
            col = pd.to_numeric(merged[tgt], errors="coerce")
            print(f"        {tgt:18s} n={col.notna().sum():3d} "
                  f"min={col.min():.3f} mean={col.mean():.3f} max={col.max():.3f}")
        else:
            print(f"        {tgt:18s} (not available)")
    if n_exp < 2:
        print(f"[14] WARNING: only {n_exp} experiment group(s). Section-level "
              f"GroupKFold needs >= 2 — add sections from more experiments.")


if __name__ == "__main__":
    main()
