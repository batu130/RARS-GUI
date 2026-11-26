"""
RARS — Rule-Based Additive Risk Scoring Studio
Scientific Nested-CV Validation & Clinical Bedside Risk Scoring GUI
Author: Selahaddin Batuhan Akben & Yavuz Selim Isler
License: MIT
"""

import os
import re
import json
import time
import threading
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import fisher_exact
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    recall_score, f1_score, matthews_corrcoef, brier_score_loss,
    roc_curve, precision_recall_curve, balanced_accuracy_score
)

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"


# ==============================================================================
# 1. EXACT 1D BINOMIAL GLM (IRLS — MATCHES STATSMODELS TO ~1e-13, ZERO BLOAT)
# ==============================================================================
def fit_binomial_glm_1d(x, y, weights=None, offset=None, max_iter=100, tol=1e-10):
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    off = np.zeros(n, dtype=float) if offset is None else np.asarray(offset, dtype=float)
    if x is None:
        X = np.ones((n, 1), dtype=float)
    else:
        X = np.column_stack([np.ones(n, dtype=float), np.asarray(x, dtype=float)])
    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(max_iter):
        eta = np.clip(X @ beta + off, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        var = np.clip(mu * (1.0 - mu), 1e-12, None)
        W = w * var
        H = X.T @ (X * W[:, None])
        g = X.T @ (w * (y - mu))
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


# ==============================================================================
# 2. UNIVERSAL CLINICAL DATA LOADER (.xlsx, .xls, .csv, .txt, .mat, .arff)
# ==============================================================================
MISSING_TOKENS = {"?", "", "na", "n/a", "nan", "null", "none", "missing", "-"}


def load_tabular_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext in (".csv", ".txt", ".data"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = "".join([f.readline() for _ in range(5)])
        sep = "\t" if "\t" in sample else (";" if sample.count(";") > sample.count(",") else ",")
        try:
            df = pd.read_csv(path, sep=sep, na_values=list(MISSING_TOKENS), keep_default_na=True)
            if df.shape[1] == 1 and sep != r"\s+":
                df = pd.read_csv(path, sep=r"\s+", engine="python", na_values=list(MISSING_TOKENS))
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python", na_values=list(MISSING_TOKENS))
    elif ext == ".arff":
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        attrs, rows, in_data = [], [], False
        for line in lines:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if s.lower().startswith("@attribute"):
                m = re.match(r"@attribute\s+'?([^'\s]+)'?\s+(.*)", s, re.I)
                if m:
                    attrs.append(m.group(1))
            elif s.lower() == "@data":
                in_data = True
            elif in_data:
                parts = [p.strip() for p in s.split(",")]
                while len(parts) > len(attrs) and parts[-1] == "":
                    parts.pop()
                if len(parts) > len(attrs):
                    parts = [p for p in parts if p != ""]
                if len(parts) == len(attrs):
                    rows.append(parts)
        df = pd.DataFrame(rows, columns=attrs)
    elif ext == ".mat":
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if not keys:
            raise ValueError("No data variables found inside .mat file.")
        # Prefer X/y pair or largest 2D matrix
        if "X" in mat and "y" in mat:
            X_arr = np.atleast_2d(np.asarray(mat["X"], dtype=float))
            y_arr = np.asarray(mat["y"]).ravel()
            cols = [f"F{i+1}" for i in range(X_arr.shape[1])]
            if "feature_names" in mat:
                fn = [str(v).strip() for v in np.atleast_1d(mat["feature_names"])]
                if len(fn) == X_arr.shape[1]:
                    cols = fn
            df = pd.DataFrame(X_arr, columns=cols)
            df["Target"] = y_arr
        else:
            best_k = None
            best_size = -1
            for k in keys:
                arr = mat[k]
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.size > best_size:
                    best_k, best_size = k, arr.size
            if best_k is None:
                # Fallback: combine 1D vectors of equal length
                vecs = {k: np.asarray(mat[k]).ravel() for k in keys if isinstance(mat[k], np.ndarray)}
                if not vecs:
                    raise ValueError("No 2D matrix or equal-length vectors found in .mat file.")
                max_len = max(len(v) for v in vecs.values())
                df = pd.DataFrame({k: v for k, v in vecs.items() if len(v) == max_len})
            else:
                arr = mat[best_k]
                cols = [f"F{i+1}" for i in range(arr.shape[1] - 1)] + ["Target"]
                df = pd.DataFrame(arr, columns=cols)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Clean whitespace and missing tokens
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda v: np.nan if (pd.isna(v) or str(v).strip().lower() in MISSING_TOKENS) else str(v).strip()
            )
            num_try = pd.to_numeric(df[col], errors="coerce")
            obs_orig = df[col].notna().sum()
            if obs_orig > 0 and num_try.notna().sum() == obs_orig:
                df[col] = num_try
    return df


def infer_column_types(X_df, max_cat_unique=6):
    num_cols, cat_cols = [], []
    for c in X_df.columns:
        s = X_df[c]
        if pd.api.types.is_numeric_dtype(s):
            uniq = s.dropna().unique()
            # Treat binary/small integer codes with <= max_cat_unique values and integer nature as categorical only if <= 2?
            # Wait: keeping integers with <= 2 unique values as categorical matches Framingham/Heart (sex, exang, male, diabetes)
            # while ordinal counts like ca/al/su/sg stay numeric unless non-numeric.
            if len(uniq) <= 2 and all(float(v).is_integer() for v in uniq):
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def encode_binary_target(series):
    s = series.dropna()
    uniq = list(pd.unique(s))
    if len(uniq) != 2:
        raise ValueError(f"Target column must be binary (2 unique classes), found {len(uniq)}: {uniq[:6]}")
    # Standard numeric 0/1 check
    try:
        nums = sorted([float(u) for u in uniq])
        if nums == [0.0, 1.0]:
            return pd.to_numeric(series, errors="coerce").astype(int).values, {0: "0", 1: "1"}
        # e.g., 1/2 or -1/1 -> larger is positive
        pos_val = nums[1]
        y = (pd.to_numeric(series, errors="coerce") == pos_val).astype(int).values
        return y, {0: str(nums[0]), 1: str(nums[1])}
    except Exception:
        s_str = series.astype(str).str.strip().str.lower()
        neg_hints = ("not", "no", "neg", "benign", "normal", "healthy", "control", "0", "false")
        u_str = sorted(list(pd.unique(s_str.dropna())))
        if any(u_str[0].startswith(h) for h in neg_hints):
            neg_lbl, pos_lbl = u_str[0], u_str[1]
        elif any(u_str[1].startswith(h) for h in neg_hints):
            neg_lbl, pos_lbl = u_str[1], u_str[0]
        else:
            neg_lbl, pos_lbl = u_str[0], u_str[1]
        y = (s_str == pos_lbl).astype(int).values
        return y, {0: neg_lbl, 1: pos_lbl}


# ==============================================================================
# 3. RARS CORE ENGINE (BIDIRECTIONAL PASSIVE RULES + 5-FOLD OOF PLATT)
# ==============================================================================
def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for b in range(n_bins):
        mask = (y_prob >= bins[b]) & (y_prob <= bins[b + 1] if b == n_bins - 1 else y_prob < bins[b + 1])
        if np.sum(mask) > 0:
            ece += (np.sum(mask) / n) * abs(np.mean(y_true[mask]) - np.mean(y_prob[mask]))
    return float(ece)


def compute_calib_slope_citl(y_true, y_prob):
    p = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    lp = np.log(p / (1.0 - p))
    try:
        slope = float(fit_binomial_glm_1d(lp, y_true)[1])
        citl = float(fit_binomial_glm_1d(None, y_true, offset=lp)[0])
    except Exception:
        slope, citl = np.nan, np.nan
    return slope, citl


def eval_metrics(y_true, y_prob, thr):
    yp = (y_prob >= thr).astype(int)
    brier = float(brier_score_loss(y_true, y_prob))
    brier_null = float(np.mean((y_true - np.mean(y_true)) ** 2))
    bss = 1.0 - (brier / brier_null) if brier_null > 1e-9 else 0.0
    auc_v = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    pr_v = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    return {
        "AUC": auc_v,
        "PR_AUC": pr_v,
        "Accuracy": float(accuracy_score(y_true, yp)),
        "BalAcc": float(balanced_accuracy_score(y_true, yp)),
        "Sensitivity": float(recall_score(y_true, yp, pos_label=1, zero_division=0)),
        "Specificity": float(recall_score(y_true, yp, pos_label=0, zero_division=0)),
        "F1": float(f1_score(y_true, yp, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, yp)),
        "Brier": brier,
        "BSS": float(bss),
        "ECE": compute_ece(y_true, y_prob),
    }


def fit_rars_rules(X_df, y, num_cols, cat_cols, alpha=1.0, cov_min=0.05, cov_max=0.95, p_thresh=0.05):
    N = len(y)
    P_tot, N_tot = np.sum(y == 1), np.sum(y == 0)
    prior_lo = float(np.log(((P_tot + alpha) / (N + 2 * alpha)) / ((N_tot + alpha) / (N + 2 * alpha))))
    rules = []

    for col_name in num_cols:
        col = pd.to_numeric(X_df[col_name], errors="coerce").values
        obs = ~np.isnan(col)
        if np.sum(obs) < 10:
            continue
        col_o, y_o = col[obs], y[obs]
        p_o, n_o = np.sum(y_o == 1), np.sum(y_o == 0)
        if p_o == 0 or n_o == 0:
            continue
        thrs = np.unique(np.percentile(col_o, np.arange(1, 100)))
        best_J, best_thr = -np.inf, None
        for t in thrs:
            gt = col_o > t
            cov = np.mean(gt)
            if cov < cov_min or cov > cov_max:
                continue
            J = abs(np.sum(y_o[gt] == 1) / p_o - np.sum(y_o[gt] == 0) / n_o)
            if J > best_J:
                best_J, best_thr = J, float(t)
        if best_thr is None:
            continue
        gt = col_o > best_thr
        ct = [[int(np.sum(y_o[gt] == 1)), int(np.sum(y_o[gt] == 0))],
              [int(np.sum(y_o[~gt] == 1)), int(np.sum(y_o[~gt] == 0))]]
        if fisher_exact(ct)[1] < p_thresh:
            for op, mask in [(">", gt), ("<=", ~gt)]:
                p1 = (np.sum(y_o[mask] == 1) + alpha) / (np.sum(mask) + 2 * alpha)
                p0 = (np.sum(y_o[mask] == 0) + alpha) / (np.sum(mask) + 2 * alpha)
                rules.append({
                    "col": col_name, "type": "num", "op": op, "thr": best_thr,
                    "w_raw": float(np.log(p1 / p0) - prior_lo),
                    "cov": float(np.mean(mask))
                })

    for col_name in cat_cols:
        col = X_df[col_name].astype(object).values
        obs = np.array([not pd.isna(v) for v in col])
        if np.sum(obs) < 10:
            continue
        col_o = np.array([_norm_cat(v) for v in col[obs]])
        y_o = y[obs]
        insig_mask = np.zeros(len(y_o), dtype=bool)
        insig_cats = []
        for cat in sorted(set(col_o)):
            m = (col_o == cat)
            cov = np.mean(m)
            if cov < cov_min or cov > cov_max:
                insig_mask |= m
                insig_cats.append(cat)
                continue
            ct = [[int(np.sum(y_o[m] == 1)), int(np.sum(y_o[m] == 0))],
                  [int(np.sum(y_o[~m] == 1)), int(np.sum(y_o[~m] == 0))]]
            if fisher_exact(ct)[1] < p_thresh:
                p1 = (np.sum(y_o[m] == 1) + alpha) / (np.sum(m) + 2 * alpha)
                p0 = (np.sum(y_o[m] == 0) + alpha) / (np.sum(m) + 2 * alpha)
                rules.append({
                    "col": col_name, "type": "cat", "op": "==", "val": cat,
                    "w_raw": float(np.log(p1 / p0) - prior_lo), "cov": float(cov)
                })
            else:
                insig_mask |= m
                insig_cats.append(cat)
        if cov_min <= np.mean(insig_mask) <= cov_max and len(insig_cats) > 0:
            ct = [[int(np.sum(y_o[insig_mask] == 1)), int(np.sum(y_o[insig_mask] == 0))],
                  [int(np.sum(y_o[~insig_mask] == 1)), int(np.sum(y_o[~insig_mask] == 0))]]
            if fisher_exact(ct)[1] < p_thresh:
                p1 = (np.sum(y_o[insig_mask] == 1) + alpha) / (np.sum(insig_mask) + 2 * alpha)
                p0 = (np.sum(y_o[insig_mask] == 0) + alpha) / (np.sum(insig_mask) + 2 * alpha)
                rules.append({
                    "col": col_name, "type": "other", "op": "in", "vals": sorted(list(set(insig_cats))),
                    "w_raw": float(np.log(p1 / p0) - prior_lo), "cov": float(np.mean(insig_mask))
                })
    return rules, prior_lo


def _norm_cat(v):
    try:
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
    except Exception:
        pass
    return str(v).strip()


def score_rars_raw(X_df, rules):
    N = len(X_df)
    s = np.zeros(N, dtype=float)
    active_cnt = np.zeros(N, dtype=int)
    for r in rules:
        if r["col"] not in X_df.columns:
            continue
        if r["type"] == "num":
            col = pd.to_numeric(X_df[r["col"]], errors="coerce").values
            obs = ~np.isnan(col)
            fire = (obs & (col > r["thr"])) if r["op"] == ">" else (obs & (col <= r["thr"]))
        elif r["type"] == "cat":
            col = X_df[r["col"]].astype(object).values
            fire = np.array([(not pd.isna(v)) and (_norm_cat(v) == r["val"]) for v in col])
        else:
            vals_set = set(r["vals"])
            col = X_df[r["col"]].astype(object).values
            fire = np.array([(not pd.isna(v)) and (_norm_cat(v) in vals_set) for v in col])
        s[fire] += r["w_raw"]
        active_cnt[fire] += 1
    return s, active_cnt


def fit_rars(X_df, y, num_cols, cat_cols, mode="oof_uw", groups=None, seed=42,
             alpha=1.0, cov_min=0.05, cov_max=0.95, p_thresh=0.05, inner_folds=5):
    rules, prior_lo = fit_rars_rules(X_df, y, num_cols, cat_cols, alpha, cov_min, cov_max, p_thresh)
    feature_meta = {}
    for c in num_cols:
        s = pd.to_numeric(X_df[c], errors="coerce").dropna()
        feature_meta[c] = {"type": "num", "min": float(s.min()) if len(s) else 0.0,
                           "max": float(s.max()) if len(s) else 1.0,
                           "median": float(s.median()) if len(s) else 0.0}
    for c in cat_cols:
        vals = sorted(list(set(_norm_cat(v) for v in X_df[c].dropna())))
        feature_meta[c] = {"type": "cat", "values": vals}

    if len(rules) == 0:
        return {"rules": [], "b0": prior_lo, "b1": 1.0, "thr": 0.5, "prior_lo": prior_lo,
                "num_cols": list(num_cols), "cat_cols": list(cat_cols), "feature_meta": feature_meta,
                "mode": mode, "params": {"alpha": alpha, "cov_min": cov_min, "cov_max": cov_max, "p_thresh": p_thresh}}

    if groups is not None:
        n_sp = max(2, min(inner_folds, len(np.unique(groups[y == 0])), len(np.unique(groups[y == 1]))))
        splits = list(StratifiedGroupKFold(n_splits=n_sp).split(X_df, y, groups))
    else:
        n_sp = max(2, min(inner_folds, int(np.sum(y == 0)), int(np.sum(y == 1))))
        splits = list(StratifiedKFold(n_splits=n_sp, shuffle=True, random_state=seed).split(X_df, y))

    oof_raw = np.zeros(len(y))
    for tri, vai in splits:
        r_in, _ = fit_rars_rules(X_df.iloc[tri], y[tri], num_cols, cat_cols, alpha, cov_min, cov_max, p_thresh)
        oof_raw[vai], _ = score_rars_raw(X_df.iloc[vai], r_in)

    if mode == "raw":
        b0, b1 = prior_lo, 1.0
    elif mode == "cw":
        s_tr, _ = score_rars_raw(X_df, rules)
        N, P_t, N_t = len(y), np.sum(y == 1), np.sum(y == 0)
        cw = np.where(y == 1, N / (2 * P_t), N / (2 * N_t))
        try:
            beta = fit_binomial_glm_1d(s_tr, y, weights=cw)
            b0, b1 = float(beta[0]), float(beta[1])
        except Exception:
            b0, b1 = prior_lo, 1.0
    else:  # oof_uw
        try:
            beta = fit_binomial_glm_1d(oof_raw, y)
            b0, b1 = float(beta[0]), float(beta[1])
        except Exception:
            b0, b1 = prior_lo, 1.0

    oof_p = 1.0 / (1.0 + np.exp(-np.clip(b0 + b1 * oof_raw, -30.0, 30.0)))
    fpr, tpr, thrs = roc_curve(y, oof_p)
    opt_thr = float(np.clip(thrs[np.argmax(tpr - fpr)], 0.02, 0.98))
    for r in rules:
        r["w_final"] = float(r["w_raw"] * b1)
        r["eg"] = float(abs(r["w_final"]) * r["cov"])
    rules.sort(key=lambda d: d["eg"], reverse=True)
    return {
        "rules": rules, "b0": b0, "b1": b1, "thr": opt_thr, "prior_lo": prior_lo,
        "num_cols": list(num_cols), "cat_cols": list(cat_cols), "feature_meta": feature_meta,
        "mode": mode, "params": {"alpha": alpha, "cov_min": cov_min, "cov_max": cov_max, "p_thresh": p_thresh}
    }


def predict_rars(X_df, model):
    s, act = score_rars_raw(X_df, model["rules"])
    p = 1.0 / (1.0 + np.exp(-np.clip(model["b0"] + model["b1"] * s, -30.0, 30.0)))
    return p, act


def score_single_patient(patient_dict, model):
    """Exact single-patient scoring with rule-by-rule audit trail."""
    tot_w = float(model["b0"])
    triggered = []
    missing_feats = []
    seen_missing = set()

    for col in model["num_cols"] + model["cat_cols"]:
        val = patient_dict.get(col, np.nan)
        if pd.isna(val) or str(val).strip() == "":
            missing_feats.append(col)
            seen_missing.add(col)

    for r in model["rules"]:
        col = r["col"]
        val = patient_dict.get(col, np.nan)
        if pd.isna(val) or str(val).strip() == "":
            continue
        fire = False
        if r["type"] == "num":
            try:
                fv = float(val)
                fire = (fv > r["thr"]) if r["op"] == ">" else (fv <= r["thr"])
            except Exception:
                continue
        elif r["type"] == "cat":
            fire = (_norm_cat(val) == r["val"])
        else:
            fire = (_norm_cat(val) in set(r["vals"]))
        if fire:
            tot_w += float(r["w_final"])
            cond = f"{col} {r['op']} {r['thr']:.4g}" if r["type"] == "num" else (
                f"{col} == {r['val']}" if r["type"] == "cat" else f"{col} in {r['vals']}"
            )
            triggered.append({"Rule": cond, "Observed": str(val), "Weight": float(r["w_final"])})

    logit = float(np.clip(tot_w, -30.0, 30.0))
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    pred = int(prob >= model["thr"])
    return {
        "prob": prob, "logit": tot_w, "pred": pred, "thr": float(model["thr"]),
        "b0": float(model["b0"]), "triggered": triggered, "missing": missing_feats
    }


# ==============================================================================
# 4. BENCHMARK COMPARATORS (INCLUDING INTSCORE)
# ==============================================================================
class IntegerScorecardClassifier:
    def __init__(self, n_bins=4, max_points=10, C=0.5, random_state=42):
        self.n_bins, self.max_points, self.C, self.random_state = n_bins, max_points, C, random_state

    def fit(self, X, y):
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=self.C, max_iter=2000, random_state=self.random_state)
        lr.fit(X, y)
        coef = lr.coef_[0]
        m = np.max(np.abs(coef))
        self.pts_ = np.round((coef / m) * self.max_points).astype(int) if m > 1e-9 else np.zeros_like(coef, dtype=int)
        raw_s = X @ self.pts_
        try:
            beta = fit_binomial_glm_1d(raw_s, y)
            self.b0_, self.b1_ = float(beta[0]), float(beta[1])
        except Exception:
            self.b0_, self.b1_ = 0.0, 0.1
        return self

    def predict_proba(self, X):
        raw_s = X @ self.pts_
        p1 = 1.0 / (1.0 + np.exp(-np.clip(self.b0_ + self.b1_ * raw_s, -30.0, 30.0)))
        return np.column_stack([1.0 - p1, p1])


def make_benchmark_pipeline(model_name, num_cols, cat_cols, seed=42):
    transformers = []
    if num_cols:
        if model_name == "GAM":
            num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                 ("sp", SplineTransformer(n_knots=4, degree=3, include_bias=False))])
        elif model_name == "IntScore":
            num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("kb", KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile"))])
        else:
            num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        transformers.append(("cat", cat_pipe, cat_cols))
    pre = ColumnTransformer(transformers)

    clf_map = {
        "DT": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=seed),
        "LR": LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        "Lasso": LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000, random_state=seed),
        "LDA": LinearDiscriminantAnalysis(),
        "NB": GaussianNB(),
        "GAM": LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        "GBT": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed),
        "IntScore": IntegerScorecardClassifier(n_bins=4, max_points=10, C=0.5, random_state=seed),
    }
    return Pipeline([("pre", pre), ("clf", clf_map[model_name])])


def run_benchmark_fold(model_name, X_tr, y_tr, X_te, num_cols, cat_cols, groups_tr=None, seed=42, inner_folds=5):
    X_tr_c = X_tr.copy()
    X_te_c = X_te.copy()
    for c in cat_cols:
        X_tr_c[c] = X_tr_c[c].map(lambda v: np.nan if pd.isna(v) else _norm_cat(v))
        X_te_c[c] = X_te_c[c].map(lambda v: np.nan if pd.isna(v) else _norm_cat(v))
    t0 = time.perf_counter()
    pipe = make_benchmark_pipeline(model_name, num_cols, cat_cols, seed)
    pipe.fit(X_tr_c, y_tr)
    t_fit = time.perf_counter() - t0

    if groups_tr is not None:
        n_sp = max(2, min(inner_folds, len(np.unique(groups_tr[y_tr == 0])), len(np.unique(groups_tr[y_tr == 1]))))
        splits = list(StratifiedGroupKFold(n_splits=n_sp).split(X_tr_c, y_tr, groups_tr))
    else:
        n_sp = max(2, min(inner_folds, int(np.sum(y_tr == 0)), int(np.sum(y_tr == 1))))
        splits = list(StratifiedKFold(n_splits=n_sp, shuffle=True, random_state=seed).split(X_tr_c, y_tr))
    oof_p = np.zeros(len(y_tr))
    for tri, vai in splits:
        p_in = make_benchmark_pipeline(model_name, num_cols, cat_cols, seed)
        p_in.fit(X_tr_c.iloc[tri], y_tr[tri])
        oof_p[vai] = p_in.predict_proba(X_tr_c.iloc[vai])[:, 1]
    fpr, tpr, thrs = roc_curve(y_tr, oof_p)
    thr = float(np.clip(thrs[np.argmax(tpr - fpr)], 0.02, 0.98))

    t1 = time.perf_counter()
    p_te = pipe.predict_proba(X_te_c)[:, 1]
    t_inf = time.perf_counter() - t1
    return p_te, thr, t_fit, t_inf


# ==============================================================================
# 5. PUBLICATION-QUALITY FIGURE GENERATOR (300 DPI)
# ==============================================================================
def build_figure(fig_type, analysis_res, dpi=110):
    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=dpi)
    y_all = analysis_res["pooled"]["RARS"]["y"]
    p_all = analysis_res["pooled"]["RARS"]["p"]
    thr_mean = float(np.mean(analysis_res["pooled"]["RARS"]["thr"]))

    if fig_type == "ROC & PR Curves":
        plt.close(fig)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 4.0), dpi=dpi)
        colors = {"RARS": "#c0392b", "LR": "#1f5a8a", "GAM": "#2e8b57", "GBT": "#7d3c98", "IntScore": "#16a085", "DT": "#e67e22"}
        for m, md in analysis_res["pooled"].items():
            if m not in colors:
                continue
            yt, yp = md["y"], md["p"]
            fpr, tpr, _ = roc_curve(yt, yp)
            auc_v = roc_auc_score(yt, yp)
            ax1.plot(fpr, tpr, color=colors[m], lw=2.2 if m == "RARS" else 1.4, label=f"{m} (AUC={auc_v:.3f})")
            prec, rec, _ = precision_recall_curve(yt, yp)
            ap_v = average_precision_score(yt, yp)
            ax2.plot(rec, prec, color=colors[m], lw=2.2 if m == "RARS" else 1.4, label=f"{m} (AP={ap_v:.3f})")
        ax1.plot([0, 1], [0, 1], "k--", lw=1.0)
        ax1.set_title("ROC Curves (Pooled Test)", fontweight="bold"); ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
        ax1.grid(True, ls="--", alpha=0.4); ax1.legend(fontsize=8)
        ax2.set_title("Precision-Recall Curves", fontweight="bold"); ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.grid(True, ls="--", alpha=0.4); ax2.legend(fontsize=8)
        fig.tight_layout()
        return fig

    elif fig_type == "Calibration (Reliability)":
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect Calibration")
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for b in range(10):
            mk = (p_all >= bins[b]) & (p_all <= bins[b + 1] if b == 9 else p_all < bins[b + 1])
            if np.sum(mk) >= 3:
                xs.append(np.mean(p_all[mk])); ys.append(np.mean(y_all[mk]))
        ax.plot(xs, ys, "-o", color="#c0392b", lw=2.2, label=f"RARS (Brier={brier_score_loss(y_all, p_all):.3f})")
        if "LR" in analysis_res["pooled"]:
            p_lr = analysis_res["pooled"]["LR"]["p"]
            xl, yl = [], []
            for b in range(10):
                mk = (p_lr >= bins[b]) & (p_lr <= bins[b + 1] if b == 9 else p_lr < bins[b + 1])
                if np.sum(mk) >= 3:
                    xl.append(np.mean(p_lr[mk])); yl.append(np.mean(y_all[mk]))
            ax.plot(xl, yl, "--s", color="#1f5a8a", lw=1.6, label=f"LR (Brier={brier_score_loss(y_all, p_lr):.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("Reliability Diagram (Pooled Out-of-Sample)", fontweight="bold")
        ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Observed Positive Rate")
        ax.grid(True, ls="--", alpha=0.4); ax.legend(fontsize=8.5)

    elif fig_type == "Risk Density & Threshold":
        ax.hist(p_all[y_all == 0], bins=20, range=(0, 1), density=True, alpha=0.55, color="#2980b9", label="Class 0")
        ax.hist(p_all[y_all == 1], bins=20, range=(0, 1), density=True, alpha=0.55, color="#c0392b", label="Class 1")
        ax.axvline(thr_mean, color="black", ls="--", lw=2.0, label=f"Threshold={thr_mean:.2f}")
        npv = 100 * np.mean(y_all[p_all < thr_mean] == 0) if np.any(p_all < thr_mean) else 0.0
        ppv = 100 * np.mean(y_all[p_all >= thr_mean] == 1) if np.any(p_all >= thr_mean) else 0.0
        ax.set_title(f"Risk Probability Density (NPV: {npv:.1f}% | PPV: {ppv:.1f}%)", fontweight="bold")
        ax.set_xlabel("Predicted Risk Probability"); ax.set_ylabel("Density (PDF)")
        ax.grid(True, ls="--", alpha=0.4); ax.legend(fontsize=8.5)

    elif fig_type == "Decision Curve Analysis (DCA)":
        pts = np.linspace(0.05, 0.80, 60)
        prev = float(np.mean(y_all))
        ax.plot(pts, prev - (1 - prev) * pts / (1 - pts), ":", color="gray", label="Treat All")
        ax.axhline(0, color="black", lw=1.0, label="Treat None")
        style_map = {"RARS": ("#c0392b", "-"), "LR": ("#1f5a8a", "--"), "IntScore": ("#16a085", ":"), "DT": ("#e67e22", "-.")}
        for m, (col, ls) in style_map.items():
            if m in analysis_res["pooled"]:
                yp = analysis_res["pooled"][m]["p"]
                nbs = [np.mean((yp >= pt) & (y_all == 1)) - np.mean((yp >= pt) & (y_all == 0)) * (pt / (1 - pt)) for pt in pts]
                ax.plot(pts, nbs, ls, color=col, lw=2.2 if m == "RARS" else 1.5, label=m)
        ax.set_ylim(-0.05, max(0.35, prev + 0.05))
        ax.set_title("Decision Curve Analysis (Net Clinical Benefit)", fontweight="bold")
        ax.set_xlabel("Threshold Probability (pt)"); ax.set_ylabel("Net Benefit")
        ax.grid(True, ls="--", alpha=0.4); ax.legend(fontsize=8.5)

    fig.tight_layout()
    return fig


# ==============================================================================
# 6. TKINTER DESKTOP GUI APPLICATION
# ==============================================================================
class RARSStudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RARS Studio — Rule-Based Additive Risk Scoring (Validation & Clinical CDSS)")
        self.geometry("1260x820")
        self.minsize(1100, 720)

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#1B365D")
        style.configure("Sub.TLabel", font=("Segoe UI", 9, "bold"), foreground="#2C3E50")
        style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"))

        # State variables
        self.df = None
        self.filepath = None
        self.analysis_res = None
        self.active_model = None
        self.target_map = {0: "Negative (0)", 1: "Positive (1)"}
        self.canvas_widget = None
        self.patient_inputs = {}
        self.ext_df = None
        self.ext_pred_df = None

        self._build_ui()

    def _build_ui(self):
        # Top Header Bar
        top = ttk.Frame(self, padding=(12, 8))
        top.pack(fill="x")
        ttk.Label(top, text="RARS Studio — Rule-Based Additive Risk Scoring", style="Header.TLabel").pack(side="left")
        ttk.Label(top, text="Bidirectional Passive Rules | 5-Fold Unweighted OOF Platt Calibration | Zero-Tuning CDSS",
                  foreground="#555555").pack(side="right")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.tab_sci = ttk.Frame(self.nb, padding=8)
        self.tab_clin = ttk.Frame(self.nb, padding=8)
        self.nb.add(self.tab_sci, text="  1. Scientific Validation (Nested CV & Benchmarks)  ")
        self.nb.add(self.tab_clin, text="  2. Clinical Bedside Scoring & External Test  ")

        self._build_scientific_tab()
        self._build_clinical_tab()

    # ------------------------------------------------------------------
    # TAB 1: SCIENTIFIC VALIDATION
    # ------------------------------------------------------------------
    def _build_scientific_tab(self):
        left = ttk.Frame(self.tab_sci, width=360)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        right = ttk.Frame(self.tab_sci)
        right.pack(side="right", fill="both", expand=True)

        # Dataset Box
        f_data = ttk.LabelFrame(left, text="1. Dataset Input (.xlsx, .xls, .csv, .txt, .mat, .arff)", padding=8)
        f_data.pack(fill="x", pady=(0, 8))
        ttk.Button(f_data, text="Load Dataset File...", command=self.load_dataset, style="Accent.TButton").pack(fill="x")
        self.lbl_file = ttk.Label(f_data, text="No file loaded", foreground="#666666", wraplength=320)
        self.lbl_file.pack(anchor="w", pady=(4, 6))

        g1 = ttk.Frame(f_data)
        g1.pack(fill="x")
        ttk.Label(g1, text="Target Column:").grid(row=0, column=0, sticky="w", pady=2)
        self.cmb_target = ttk.Combobox(g1, state="readonly", width=22)
        self.cmb_target.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=2)

        ttk.Label(g1, text="Group/Subject ID:").grid(row=1, column=0, sticky="w", pady=2)
        self.cmb_group = ttk.Combobox(g1, state="readonly", width=22)
        self.cmb_group.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=2)
        g1.columnconfigure(1, weight=1)

        # Parameters Box
        f_par = ttk.LabelFrame(left, text="2. RARS & Nested CV Parameters", padding=8)
        f_par.pack(fill="x", pady=(0, 8))

        self.var_alpha = tk.DoubleVar(value=1.0)
        self.var_covmin = tk.DoubleVar(value=0.05)
        self.var_covmax = tk.DoubleVar(value=0.95)
        self.var_pthresh = tk.DoubleVar(value=0.05)
        self.var_calib = tk.StringVar(value="oof_uw")
        self.var_proto = tk.StringVar(value="Stratified K-Fold CV")
        self.var_outer = tk.IntVar(value=5)
        self.var_inner = tk.IntVar(value=5)
        self.var_seed = tk.IntVar(value=42)
        self.var_bench = tk.BooleanVar(value=True)

        params_grid = [
            ("Laplace alpha:", self.var_alpha, 0),
            ("Min Coverage:", self.var_covmin, 1),
            ("Max Coverage:", self.var_covmax, 2),
            ("Fisher Filter p <", self.var_pthresh, 3),
            ("Outer Folds / Reps:", self.var_outer, 6),
            ("Inner OOF Folds:", self.var_inner, 7),
            ("Random Seed:", self.var_seed, 8),
        ]
        for lbl, var, r_idx in params_grid:
            ttk.Label(f_par, text=lbl).grid(row=r_idx, column=0, sticky="w", pady=2)
            ttk.Entry(f_par, textvariable=var, width=16).grid(row=r_idx, column=1, sticky="ew", padx=(6, 0), pady=2)

        ttk.Label(f_par, text="Calibration Mode:").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Combobox(f_par, textvariable=self.var_calib, state="readonly",
                     values=["oof_uw", "cw", "raw"], width=16).grid(row=4, column=1, sticky="ew", padx=(6, 0), pady=2)

        ttk.Label(f_par, text="Validation Protocol:").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Combobox(f_par, textvariable=self.var_proto, state="readonly",
                     values=["Stratified K-Fold CV", "Repeated Hold-Out (90/10)"], width=18).grid(row=5, column=1, sticky="ew", padx=(6, 0), pady=2)

        ttk.Checkbutton(f_par, text="Include 8 Reference Models (DT, LR, Lasso, LDA, NB, GAM, GBT, IntScore)",
                        variable=self.var_bench).grid(row=9, column=0, columnspan=2, sticky="w", pady=(6, 2))
        f_par.columnconfigure(1, weight=1)

        # Action Buttons
        f_act = ttk.LabelFrame(left, text="3. Execution & Export", padding=8)
        f_act.pack(fill="x", pady=(0, 8))
        self.btn_run = ttk.Button(f_act, text="Run Scientific Analysis", command=self.start_analysis, style="Accent.TButton")
        self.btn_run.pack(fill="x", pady=(0, 6))
        self.progress = ttk.Progressbar(f_act, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 6))

        b_row = ttk.Frame(f_act)
        b_row.pack(fill="x")
        ttk.Button(b_row, text="Export Excel", command=self.export_excel).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(b_row, text="Save 300-DPI Figs", command=self.export_figures).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(b_row, text="Save Model (.json)", command=self.save_model_json).pack(side="left", expand=True, fill="x", padx=(3, 0))

        # Log Box
        f_log = ttk.LabelFrame(left, text="Execution Log", padding=6)
        f_log.pack(fill="both", expand=True)
        self.txt_log = tk.Text(f_log, height=8, font=("Consolas", 9), state="disabled", wrap="word")
        self.txt_log.pack(fill="both", expand=True)

        # Right side: Metrics Table (top) + Plot Viewer (bottom)
        f_tbl = ttk.LabelFrame(right, text="Cross-Validation / Hold-Out Performance Summary", padding=6)
        f_tbl.pack(fill="x", pady=(0, 8))
        cols = ("Model", "AUC", "PR-AUC", "Acc", "Sens", "Spec", "MCC", "Brier", "ECE", "Slope", "CITL", "Inf(ms)")
        self.tree_metrics = ttk.Treeview(f_tbl, columns=cols, show="headings", height=9)
        for c in cols:
            self.tree_metrics.heading(c, text=c)
            self.tree_metrics.column(c, width=68 if c != "Model" else 92, anchor="center")
        self.tree_metrics.pack(fill="x")

        f_plot = ttk.LabelFrame(right, text="Diagnostic Figures", padding=6)
        f_plot.pack(fill="both", expand=True)
        p_ctrl = ttk.Frame(f_plot)
        p_ctrl.pack(fill="x", pady=(0, 4))
        ttk.Label(p_ctrl, text="Select Figure:").pack(side="left")
        self.cmb_fig = ttk.Combobox(
            p_ctrl, state="readonly", width=30,
            values=["ROC & PR Curves", "Calibration (Reliability)", "Risk Density & Threshold", "Decision Curve Analysis (DCA)"]
        )
        self.cmb_fig.current(0)
        self.cmb_fig.pack(side="left", padx=8)
        self.cmb_fig.bind("<<ComboboxSelected>>", lambda e: self.refresh_plot())
        self.plot_holder = ttk.Frame(f_plot)
        self.plot_holder.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # TAB 2: CLINICAL BEDSIDE SCORING & EXTERNAL TEST
    # ------------------------------------------------------------------
    def _build_clinical_tab(self):
        # Top control bar
        bar = ttk.Frame(self.tab_clin, padding=(0, 0, 0, 8))
        bar.pack(fill="x")
        ttk.Button(bar, text="Train Model from Loaded Dataset", command=self.train_clinical_model, style="Accent.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Load Saved Model (.json)", command=self.load_model_json).pack(side="left", padx=6)
        self.lbl_model_info = ttk.Label(bar, text="No active RARS model. Train in Tab 1 or load a .json model.", foreground="#8B0000")
        self.lbl_model_info.pack(side="left", padx=12)

        paned = ttk.PanedWindow(self.tab_clin, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # Left pane: Single-patient bedside entry form
        f_pat = ttk.LabelFrame(paned, text="A. Single-Patient Bedside Calculator (Leave Unknown Fields Blank = Neutral 0)", padding=8)
        paned.add(f_pat, weight=2)

        # Scrollable form
        canvas = tk.Canvas(f_pat, highlightthickness=0)
        sb = ttk.Scrollbar(f_pat, orient="vertical", command=canvas.yview)
        self.pat_form_frame = ttk.Frame(canvas)
        self.pat_form_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.pat_form_frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Bottom controls for single patient
        f_pat_btn = ttk.Frame(self.tab_clin)
        # Put calculate button inside a right pane top box so it's always visible!
        right_p = ttk.Frame(paned)
        paned.add(right_p, weight=3)

        # Risk Output Panel
        f_out = ttk.LabelFrame(right_p, text="Patient Risk Assessment & Rule Audit Trail", padding=8)
        f_out.pack(fill="both", expand=True, pady=(0, 8))

        btn_row = ttk.Frame(f_out)
        btn_row.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_row, text="Calculate Patient Risk Score", command=self.calc_patient_risk, style="Accent.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Clear All Fields (All Missing)", command=self.clear_patient_fields).pack(side="left", padx=6)

        self.lbl_risk_banner = ttk.Label(f_out, text="Risk Probability: —   |   Decision: —", font=("Segoe UI", 12, "bold"), foreground="#1B365D")
        self.lbl_risk_banner.pack(anchor="w", pady=(2, 4))
        self.lbl_risk_detail = ttk.Label(f_out, text="Logit: —  |  Active Rules: —  |  Missing (Passive Neutral): —", foreground="#444444")
        self.lbl_risk_detail.pack(anchor="w", pady=(0, 6))

        cols_tr = ("Rule Condition", "Observed Value", "Calibrated Weight (w_final)")
        self.tree_audit = ttk.Treeview(f_out, columns=cols_tr, show="headings", height=8)
        for c, w in zip(cols_tr, [220, 110, 150]):
            self.tree_audit.heading(c, text=c)
            self.tree_audit.column(c, width=w, anchor="center")
        self.tree_audit.pack(fill="both", expand=True)

        # Scorecard & External Batch Test Tabs inside bottom-right
        sub_nb = ttk.Notebook(right_p)
        sub_nb.pack(fill="both", expand=True)

        tab_sc = ttk.Frame(sub_nb, padding=6)
        tab_ext = ttk.Frame(sub_nb, padding=6)
        sub_nb.add(tab_sc, text="Active RARS Scorecard Rules")
        sub_nb.add(tab_ext, text="B. Batch External Dataset Test")

        cols_sc = ("Rank", "Rule", "Coverage", "w_raw", "w_final", "Gain (E_k)")
        self.tree_scorecard = ttk.Treeview(tab_sc, columns=cols_sc, show="headings", height=9)
        for c, w in zip(cols_sc, [45, 210, 75, 75, 75, 75]):
            self.tree_scorecard.heading(c, text=c)
            self.tree_scorecard.column(c, width=w, anchor="center")
        self.tree_scorecard.pack(fill="both", expand=True)

        # Batch external validation UI
        ext_bar = ttk.Frame(tab_ext)
        ext_bar.pack(fill="x", pady=(0, 6))
        ttk.Button(ext_bar, text="Load External File...", command=self.load_external_file).pack(side="left", padx=(0, 6))
        ttk.Label(ext_bar, text="Optional True Target:").pack(side="left")
        self.cmb_ext_target = ttk.Combobox(ext_bar, state="readonly", width=18)
        self.cmb_ext_target.pack(side="left", padx=6)
        ttk.Button(ext_bar, text="Run External Test", command=self.run_external_batch, style="Accent.TButton").pack(side="left", padx=4)
        ttk.Button(ext_bar, text="Export Predictions", command=self.export_external_preds).pack(side="left", padx=4)

        self.txt_ext = tk.Text(tab_ext, height=9, font=("Consolas", 9), state="disabled", wrap="word")
        self.txt_ext.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # LOGGING & FILE LOADING
    # ------------------------------------------------------------------
    def log_msg(self, text):
        ts = time.strftime("%H:%M:%S")
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", f"[{ts}] {text}\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def load_dataset(self):
        path = filedialog.askopenfilename(
            title="Select Clinical Dataset",
            filetypes=[("Supported Data", "*.xlsx *.xls *.csv *.txt *.data *.arff *.mat"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            df = load_tabular_file(path)
            self.df = df
            self.filepath = path
            miss_pct = 100.0 * df.isna().sum().sum() / max(1, df.size)
            self.lbl_file.configure(text=f"{os.path.basename(path)} | n={len(df)}, cols={df.shape[1]}, missing={miss_pct:.2f}%", foreground="#1B365D")
            cols = list(df.columns)
            self.cmb_target["values"] = cols
            self.cmb_target.current(len(cols) - 1)
            self.cmb_group["values"] = ["(None)"] + cols
            self.cmb_group.current(0)
            self.log_msg(f"Loaded {os.path.basename(path)}: {len(df)} rows, {df.shape[1]} columns ({miss_pct:.2f}% missing).")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # ------------------------------------------------------------------
    # TAB 1: SCIENTIFIC ANALYSIS WORKER
    # ------------------------------------------------------------------
    def start_analysis(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        self.btn_run.configure(state="disabled")
        self.progress["value"] = 0
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()

    def _run_analysis_thread(self):
        try:
            target_col = self.cmb_target.get()
            group_col = self.cmb_group.get()
            df_work = self.df.dropna(subset=[target_col]).reset_index(drop=True)
            y, tmap = encode_binary_target(df_work[target_col])
            self.target_map = tmap

            drop_cols = [target_col]
            groups = None
            if group_col and group_col != "(None)" and group_col != target_col:
                groups = df_work[group_col].astype(str).values
                drop_cols.append(group_col)

            X = df_work.drop(columns=drop_cols).copy()
            num_cols, cat_cols = infer_column_types(X)

            alpha = float(self.var_alpha.get())
            cov_min = float(self.var_covmin.get())
            cov_max = float(self.var_covmax.get())
            p_thresh = float(self.var_pthresh.get())
            calib_mode = self.var_calib.get()
            proto = self.var_proto.get()
            n_outer = max(2, int(self.var_outer.get()))
            n_inner = max(2, int(self.var_inner.get()))
            seed = int(self.var_seed.get())
            do_bench = bool(self.var_bench.get())

            models_list = ["RARS"] + (["IntScore", "DT", "LR", "Lasso", "LDA", "NB", "GAM", "GBT"] if do_bench else [])

            if groups is not None:
                n_sp = max(2, min(n_outer, len(np.unique(groups[y == 0])), len(np.unique(groups[y == 1]))))
                splits = list(StratifiedGroupKFold(n_splits=n_sp).split(X, y, groups))
                proto_used = f"StratifiedGroupKFold ({n_sp} folds, {len(np.unique(groups))} groups)"
            elif proto == "Stratified K-Fold CV":
                splits = list(StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed).split(X, y))
                proto_used = f"Stratified {n_outer}-Fold CV"
            else:
                splits = [next(StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed + i).split(X, y))
                          for i in range(n_outer)]
                proto_used = f"{n_outer}x Repeated 90/10 Hold-Out"

            self.after(0, lambda: self.log_msg(
                f"Running {proto_used} | n={len(y)}, num={len(num_cols)}, cat={len(cat_cols)}, mode={calib_mode}..."
            ))

            m_store = {m: [] for m in models_list}
            t_store = {m: [] for m in models_list}
            pooled = {m: {"y": [], "p": [], "thr": []} for m in models_list}

            for fi, (tri, tei) in enumerate(splits):
                X_tr, y_tr = X.iloc[tri], y[tri]
                X_te, y_te = X.iloc[tei], y[tei]
                g_tr = groups[tri] if groups is not None else None

                t0 = time.perf_counter()
                mod = fit_rars(X_tr, y_tr, num_cols, cat_cols, mode=calib_mode, groups=g_tr, seed=seed + fi,
                               alpha=alpha, cov_min=cov_min, cov_max=cov_max, p_thresh=p_thresh, inner_folds=n_inner)
                t_fit = time.perf_counter() - t0
                t1 = time.perf_counter()
                p_te, _ = predict_rars(X_te, mod)
                t_inf = time.perf_counter() - t1

                m_store["RARS"].append(eval_metrics(y_te, p_te, mod["thr"]))
                t_store["RARS"].append((t_fit, t_inf))
                pooled["RARS"]["y"].append(y_te); pooled["RARS"]["p"].append(p_te); pooled["RARS"]["thr"].append(mod["thr"])

                for bm in models_list[1:]:
                    pb, thrb, tfb, tib = run_benchmark_fold(bm, X_tr, y_tr, X_te, num_cols, cat_cols, g_tr, seed + fi, n_inner)
                    m_store[bm].append(eval_metrics(y_te, pb, thrb))
                    t_store[bm].append((tfb, tib))
                    pooled[bm]["y"].append(y_te); pooled[bm]["p"].append(pb); pooled[bm]["thr"].append(thrb)

                pct = 100.0 * (fi + 1) / (len(splits) + 1)
                self.after(0, lambda v=pct, k=fi+1, tot=len(splits): (
                    self.progress.configure(value=v),
                    self.log_msg(f"Completed split {k}/{tot}")
                ))

            # Fit final deployment model on full dataset
            full_model = fit_rars(X, y, num_cols, cat_cols, mode=calib_mode, groups=groups, seed=seed,
                                  alpha=alpha, cov_min=cov_min, cov_max=cov_max, p_thresh=p_thresh, inner_folds=n_inner)
            full_model["target_col"] = target_col
            full_model["target_map"] = tmap

            summary = {}
            pooled_cat = {}
            for m in models_list:
                yt_c = np.concatenate(pooled[m]["y"])
                yp_c = np.concatenate(pooled[m]["p"])
                sl, ci = compute_calib_slope_citl(yt_c, yp_c)
                pooled_cat[m] = {"y": yt_c, "p": yp_c, "thr": pooled[m]["thr"]}
                agg = {}
                for k in m_store[m][0].keys():
                    vals = [d[k] for d in m_store[m]]
                    agg[k] = (float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0)
                agg["Slope"] = sl
                agg["CITL"] = ci
                agg["TrainTime"] = float(np.mean([x[0] for x in t_store[m]]))
                agg["TestTime_ms"] = float(1000.0 * np.mean([x[1] for x in t_store[m]]))
                summary[m] = agg

            self.analysis_res = {
                "summary": summary, "pooled": pooled_cat, "folds": m_store,
                "protocol": proto_used, "full_model": full_model
            }
            self.active_model = full_model

            self.after(0, self._on_analysis_done)
        except Exception as e:
            self.after(0, lambda err=str(e): (
                messagebox.showerror("Analysis Error", err),
                self.btn_run.configure(state="normal")
            ))

    def _on_analysis_done(self):
        self.progress.configure(value=100)
        self.btn_run.configure(state="normal")

        for item in self.tree_metrics.get_children():
            self.tree_metrics.delete(item)
        for m, agg in self.analysis_res["summary"].items():
            row = (
                m,
                f"{agg['AUC'][0]:.3f} ({agg['AUC'][1]:.3f})",
                f"{agg['PR_AUC'][0]:.3f}",
                f"{agg['Accuracy'][0]:.3f}",
                f"{agg['Sensitivity'][0]:.3f}",
                f"{agg['Specificity'][0]:.3f}",
                f"{agg['MCC'][0]:.3f}",
                f"{agg['Brier'][0]:.3f}",
                f"{agg['ECE'][0]:.3f}",
                f"{agg['Slope']:.2f}" if not np.isnan(agg["Slope"]) else "—",
                f"{agg['CITL']:+.2f}" if not np.isnan(agg["CITL"]) else "—",
                f"{agg['TestTime_ms']:.2f}"
            )
            self.tree_metrics.insert("", "end", values=row)

        self.refresh_plot()
        self._load_active_model_into_clinical_tab()
        self.log_msg(f"DONE! Full RARS model ({len(self.active_model['rules'])} rules) synced to Clinical Tab.")

    def refresh_plot(self):
        if self.analysis_res is None:
            return
        if self.canvas_widget is not None:
            self.canvas_widget.get_tk_widget().destroy()
        fig = build_figure(self.cmb_fig.get(), self.analysis_res, dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_widget = canvas

    def export_excel(self):
        if self.analysis_res is None:
            messagebox.showwarning("No Results", "Run analysis first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Workbook", "*.xlsx")])
        if not path:
            return
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as w:
                rows = []
                for m, agg in self.analysis_res["summary"].items():
                    rows.append({
                        "Model": m, "AUC_mean": agg["AUC"][0], "AUC_sd": agg["AUC"][1],
                        "PR_AUC": agg["PR_AUC"][0], "Accuracy": agg["Accuracy"][0],
                        "BalAcc": agg["BalAcc"][0], "Sensitivity": agg["Sensitivity"][0],
                        "Specificity": agg["Specificity"][0], "F1": agg["F1"][0],
                        "MCC": agg["MCC"][0], "Brier": agg["Brier"][0], "BSS": agg["BSS"][0],
                        "ECE": agg["ECE"][0], "Slope": agg["Slope"], "CITL": agg["CITL"],
                        "Train_s": agg["TrainTime"], "Inf_ms": agg["TestTime_ms"]
                    })
                pd.DataFrame(rows).to_excel(w, sheet_name="Summary_Metrics", index=False)
                pd.DataFrame(self.active_model["rules"]).to_excel(w, sheet_name="RARS_Scorecard", index=False)
                pd.DataFrame(self.analysis_res["folds"]["RARS"]).to_excel(w, sheet_name="RARS_Fold_Details", index=False)
            self.log_msg(f"Exported Excel report to {os.path.basename(path)}")
            messagebox.showinfo("Exported", f"Excel report saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def export_figures(self):
        if self.analysis_res is None:
            messagebox.showwarning("No Results", "Run analysis first.")
            return
        folder = filedialog.askdirectory(title="Select Folder for 300-DPI Figures")
        if not folder:
            return
        try:
            names = [
                ("ROC & PR Curves", "rars_roc_pr_300dpi.png"),
                ("Calibration (Reliability)", "rars_calibration_300dpi.png"),
                ("Risk Density & Threshold", "rars_density_threshold_300dpi.png"),
                ("Decision Curve Analysis (DCA)", "rars_dca_300dpi.png"),
            ]
            for ftype, fname in names:
                fig = build_figure(ftype, self.analysis_res, dpi=300)
                fig.savefig(os.path.join(folder, fname), dpi=300, bbox_inches="tight")
                plt.close(fig)
            self.log_msg(f"Saved 4 figures (300 dpi) to {folder}")
            messagebox.showinfo("Saved", f"4 high-resolution (300 dpi) PNG figures saved to:\n{folder}")
        except Exception as e:
            messagebox.showerror("Figure Error", str(e))

    def save_model_json(self):
        if self.active_model is None:
            messagebox.showwarning("No Model", "Train a RARS model first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("RARS Model JSON", "*.json")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.active_model, f, indent=2, ensure_ascii=False)
        self.log_msg(f"Saved RARS model to {os.path.basename(path)}")
        messagebox.showinfo("Saved", f"Model saved:\n{path}")

    # ------------------------------------------------------------------
    # TAB 2: CLINICAL SCORING & EXTERNAL VALIDATION
    # ------------------------------------------------------------------
    def train_clinical_model(self):
        if self.df is None:
            self.load_dataset()
            if self.df is None:
                return
        try:
            target_col = self.cmb_target.get() or self.df.columns[-1]
            group_col = self.cmb_group.get()
            df_w = self.df.dropna(subset=[target_col]).reset_index(drop=True)
            y, tmap = encode_binary_target(df_w[target_col])
            drop_cols = [target_col]
            groups = None
            if group_col and group_col != "(None)" and group_col != target_col:
                groups = df_w[group_col].astype(str).values
                drop_cols.append(group_col)
            X = df_w.drop(columns=drop_cols).copy()
            nc, cc = infer_column_types(X)
            mod = fit_rars(
                X, y, nc, cc, mode=self.var_calib.get(), groups=groups, seed=int(self.var_seed.get()),
                alpha=float(self.var_alpha.get()), cov_min=float(self.var_covmin.get()),
                cov_max=float(self.var_covmax.get()), p_thresh=float(self.var_pthresh.get()),
                inner_folds=int(self.var_inner.get())
            )
            mod["target_col"] = target_col
            mod["target_map"] = tmap
            self.active_model = mod
            self._load_active_model_into_clinical_tab()
            messagebox.showinfo("Model Ready", f"RARS model trained ({len(mod['rules'])} rules, threshold={mod['thr']:.3f}).")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    def load_model_json(self):
        path = filedialog.askopenfilename(title="Load RARS Model (.json)", filetypes=[("JSON Model", "*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.active_model = json.load(f)
            self._load_active_model_into_clinical_tab()
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))

    def _load_active_model_into_clinical_tab(self):
        m = self.active_model
        if m is None:
            return
        self.lbl_model_info.configure(
            text=f"Active Model: {len(m['rules'])} rules | b0={m['b0']:+.3f}, b1={m['b1']:.3f} | Threshold={m['thr']:.3f} | Mode={m.get('mode','oof_uw')}",
            foreground="#1B5E20"
        )
        # Populate Scorecard tree
        for item in self.tree_scorecard.get_children():
            self.tree_scorecard.delete(item)
        for i, r in enumerate(m["rules"], 1):
            cond = f"{r['col']} {r['op']} {r['thr']:.4g}" if r["type"] == "num" else (
                f"{r['col']} == {r['val']}" if r["type"] == "cat" else f"{r['col']} in {r['vals']}"
            )
            self.tree_scorecard.insert("", "end", values=(
                i, cond, f"{r['cov']:.3f}", f"{r['w_raw']:+.3f}", f"{r['w_final']:+.3f}", f"{r['eg']:.3f}"
            ))

        # Rebuild patient entry form dynamically
        for w in self.pat_form_frame.winfo_children():
            w.destroy()
        self.patient_inputs = {}
        all_feats = m["num_cols"] + m["cat_cols"]
        meta = m.get("feature_meta", {})
        for idx, col in enumerate(all_feats):
            info = meta.get(col, {"type": "num"})
            is_num = (info["type"] == "num")
            hint = f"[{info['min']:.3g}..{info['max']:.3g}]" if is_num and "min" in info else "(cat)"
            ttk.Label(self.pat_form_frame, text=f"{col} {hint}:").grid(row=idx, column=0, sticky="w", padx=4, pady=3)
            var = tk.StringVar(value="")
            if is_num:
                ent = ttk.Entry(self.pat_form_frame, textvariable=var, width=18)
                ent.grid(row=idx, column=1, sticky="ew", padx=4, pady=3)
            else:
                vals = [""] + [str(v) for v in info.get("values", [])]
                cb = ttk.Combobox(self.pat_form_frame, textvariable=var, values=vals, width=16)
                cb.grid(row=idx, column=1, sticky="ew", padx=4, pady=3)
            self.patient_inputs[col] = var

    def clear_patient_fields(self):
        for var in self.patient_inputs.values():
            var.set("")

    def calc_patient_risk(self):
        if self.active_model is None:
            messagebox.showwarning("No Model", "Train or load a RARS model first.")
            return
        pat = {}
        for col, var in self.patient_inputs.items():
            val = var.get().strip()
            pat[col] = np.nan if val == "" else val
        res_p = score_single_patient(pat, self.active_model)
        tmap = self.active_model.get("target_map", {"0": "Low Risk", "1": "High Risk"})
        lbl_txt = "HIGH RISK (Positive)" if res_p["pred"] == 1 else "LOW RISK (Negative)"
        col_fg = "#B71C1C" if res_p["pred"] == 1 else "#1B5E20"
        self.lbl_risk_banner.configure(
            text=f"Risk Probability: {100.0 * res_p['prob']:.2f}%   |   Decision: {lbl_txt}  (Threshold: {100.0 * res_p['thr']:.1f}%)",
            foreground=col_fg
        )
        self.lbl_risk_detail.configure(
            text=f"Intercept b0={res_p['b0']:+.3f}  |  Total Logit={res_p['logit']:+.3f}  |  Active Rules: {len(res_p['triggered'])}  |  Missing (Neutral 0): {len(res_p['missing'])}"
        )
        for item in self.tree_audit.get_children():
            self.tree_audit.delete(item)
        for tr in res_p["triggered"]:
            self.tree_audit.insert("", "end", values=(tr["Rule"], tr["Observed"], f"{tr['Weight']:+.4f}"))
        for mf in res_p["missing"]:
            self.tree_audit.insert("", "end", values=(f"{mf} [MISSING]", "NaN", "0.0000 (Passive Neutral)"))

    def load_external_file(self):
        path = filedialog.askopenfilename(
            title="Select External Validation Dataset",
            filetypes=[("Supported Data", "*.xlsx *.xls *.csv *.txt *.data *.arff *.mat"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            self.ext_df = load_tabular_file(path)
            self.cmb_ext_target["values"] = ["(No Ground Truth)"] + list(self.ext_df.columns)
            self.cmb_ext_target.current(0)
            self._write_ext_log(f"Loaded external file: {os.path.basename(path)} (n={len(self.ext_df)}, cols={self.ext_df.shape[1]})")
        except Exception as e:
            messagebox.showerror("External File Error", str(e))

    def _write_ext_log(self, msg, clear=False):
        self.txt_ext.configure(state="normal")
        if clear:
            self.txt_ext.delete("1.0", "end")
        self.txt_ext.insert("end", msg + "\n")
        self.txt_ext.see("end")
        self.txt_ext.configure(state="disabled")

    def run_external_batch(self):
        if self.active_model is None:
            messagebox.showwarning("No Model", "Train or load a RARS model first.")
            return
        if self.ext_df is None:
            messagebox.showwarning("No External File", "Load an external dataset first.")
            return
        try:
            df_e = self.ext_df.copy()
            probs, acts = predict_rars(df_e, self.active_model)
            preds = (probs >= self.active_model["thr"]).astype(int)
            out_df = df_e.copy()
            out_df["RARS_Probability"] = np.round(probs, 5)
            out_df["RARS_Prediction"] = preds
            out_df["RARS_Active_Rules"] = acts
            self.ext_pred_df = out_df

            lines = [
                f"=== BATCH EXTERNAL SCORING COMPLETED (n={len(df_e)}) ===",
                f"Predicted High Risk (1): {int(preds.sum())} ({100*preds.mean():.1f}%)",
                f"Mean Active Rules/Patient: {np.mean(acts):.2f}",
                f"Decision Threshold: {self.active_model['thr']:.4f}"
            ]
            tcol = self.cmb_ext_target.get()
            if tcol and tcol != "(No Ground Truth)" and tcol in df_e.columns:
                mask = df_e[tcol].notna().values
                y_true, _ = encode_binary_target(df_e.loc[mask, tcol])
                ev = eval_metrics(y_true, probs[mask], self.active_model["thr"])
                sl, ci = compute_calib_slope_citl(y_true, probs[mask])
                lines.append("\n--- External Validation Metrics ---")
                lines.append(f"AUC={ev['AUC']:.4f} | PR-AUC={ev['PR_AUC']:.4f} | Accuracy={ev['Accuracy']:.4f} | BalAcc={ev['BalAcc']:.4f}")
                lines.append(f"Sens={ev['Sensitivity']:.4f} | Spec={ev['Specificity']:.4f} | F1={ev['F1']:.4f} | MCC={ev['MCC']:.4f}")
                lines.append(f"Brier={ev['Brier']:.4f} | BSS={ev['BSS']:.4f} | ECE={ev['ECE']:.4f} | Slope={sl:.3f} | CITL={ci:+.3f}")
            self._write_ext_log("\n".join(lines), clear=True)
        except Exception as e:
            messagebox.showerror("External Scoring Error", str(e))

    def export_external_preds(self):
        if self.ext_pred_df is None:
            messagebox.showwarning("No Predictions", "Run External Test first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")])
        if not path:
            return
        if path.lower().endswith(".csv"):
            self.ext_pred_df.to_csv(path, index=False)
        else:
            self.ext_pred_df.to_excel(path, index=False)
        messagebox.showinfo("Exported", f"Predictions saved to:\n{path}")


if __name__ == "__main__":
    app = RARSStudioApp()
    app.mainloop()
