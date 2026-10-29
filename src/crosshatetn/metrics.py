"""Métriques de classification, intervalles bootstrap et bootstrap apparié."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def scores(y_true, y_pred, labels: list[str]) -> dict:
    """macro-F1 (métrique principale), exactitude et précision/rappel/F1 par classe."""
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {"macro_f1": float(np.mean(f)), "accuracy": float(accuracy_score(y_true, y_pred))}
    for i, lab in enumerate(labels):
        out[f"precision_{lab}"], out[f"recall_{lab}"], out[f"f1_{lab}"] = float(p[i]), float(r[i]), float(f[i])
    return out


def _encode(y_true, preds: list, labels: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
    index = {lab: i for i, lab in enumerate(labels)}
    enc = lambda y: np.array([index[v] for v in y], dtype=np.int64)  # noqa: E731
    return enc(y_true), [enc(p) for p in preds]


def _macro_f1_batch(yt: np.ndarray, yp: np.ndarray, k: int) -> np.ndarray:
    """macro-F1 de plusieurs rééchantillonnages à la fois (lignes de yt, yp), via les matrices de confusion."""
    b = yt.shape[0]
    codes = (yt * k + yp) + (np.arange(b)[:, None] * k * k)
    conf = np.bincount(codes.ravel(), minlength=b * k * k).reshape(b, k, k)
    tp = np.diagonal(conf, axis1=1, axis2=2)
    denom = conf.sum(axis=1) + conf.sum(axis=2)  # (tp + fn) + (tp + fp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return f1.mean(axis=1)


def _resample_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(y_true, y_pred, labels: list[str], n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """macro-F1 et intervalle de confiance bootstrap (percentiles) à 1 - alpha."""
    yt, (yp,) = _encode(y_true, [y_pred], labels)
    point = float(_macro_f1_batch(yt[None, :], yp[None, :], len(labels))[0])
    idx = _resample_indices(len(yt), n_boot, seed)
    boot = _macro_f1_batch(yt[idx], yp[idx], len(labels))
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return point, float(low), float(high)


def paired_bootstrap(y_true, pred_a, pred_b, labels: list[str], n_boot: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Compare deux systèmes sur les mêmes exemples (mêmes rééchantillonnages pour les deux).

    Renvoie la différence de macro-F1 (A - B), son intervalle de confiance et une p-valeur bilatérale.
    """
    yt, (pa, pb) = _encode(y_true, [pred_a, pred_b], labels)
    k = len(labels)
    diff = float(_macro_f1_batch(yt[None], pa[None], k)[0] - _macro_f1_batch(yt[None], pb[None], k)[0])
    idx = _resample_indices(len(yt), n_boot, seed)
    boot = _macro_f1_batch(yt[idx], pa[idx], k) - _macro_f1_batch(yt[idx], pb[idx], k)
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p_value = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return {"diff_macro_f1": diff, "ic_bas": float(low), "ic_haut": float(high), "p_value": float(p_value)}


def confusion_table(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    table = pd.crosstab(pd.Series(y_true, name="vrai"), pd.Series(y_pred, name="prédit"))
    return table.reindex(index=labels, columns=labels, fill_value=0)
