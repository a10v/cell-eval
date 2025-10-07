"""DE metrics module."""

from typing import Literal

import polars as pl
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from .._types import DEComparison, DESortBy
import numpy as np

def de_overlap_metric(
    data: DEComparison,
    k: int | None,
    metric: Literal["precision", "overlap"] = "overlap",
    fdr_threshold: float = 0.05,
    sort_by: DESortBy = DESortBy.ABS_FOLD_CHANGE,
) -> dict[str, float]:
    """Compute overlap between real and predicted DE genes.

    Note: use `k` argument for measuring recall and use `topk` argument for measuring precision.

    """
    return data.compute_overlap(
        k=k,
        metric=metric,
        fdr_threshold=fdr_threshold,
        sort_by=sort_by,
    )

class DEWeightedSpearmanLFC:
    """Weighted Spearman on LFC; optionally scaled by dataset overlap metric."""

    def __init__(
        self,
        fdr_threshold: float = 0.05,
        gamma: float = 1.0,
        overlap_metric: Literal["precision", "overlap"] = "overlap",
        k: int | None = None,            # top-k for overlap/precision if you want it
        sort_by=None,                    # e.g., DESortBy.ABS_FOLD_CHANGE
        use_overlap: bool = True,        # turn off to use plain weighted Spearman
    ) -> None:
        self.fdr_threshold = fdr_threshold
        self.gamma = gamma
        self.overlap_metric = overlap_metric
        self.k = k
        self.sort_by = sort_by
        self.use_overlap = use_overlap

    def __call__(self, data) -> dict[str, float]:
        # Precompute per-pert overlap once (if requested and available)
        overlap = {}
        if self.use_overlap and hasattr(data, "compute_overlap"):
            try:
                overlap = data.compute_overlap(
                    k=self.k,
                    metric=self.overlap_metric,
                    fdr_threshold=self.fdr_threshold,
                    sort_by=self.sort_by,
                )
            except Exception:
                overlap = {}

        correlations: dict[str, float] = {}

        merged = data.real.filter_to_significant(fdr_threshold=self.fdr_threshold).join(
            data.pred.data,
            on=[data.real.target_col, data.real.feature_col],
            suffix="_pred",
            how="inner",
        )

        # iterate per perturbation
        for pert, df in merged.group_by(data.real.target_col):
            l_true = df[data.real.fold_change_col].to_numpy()
            l_pred = df[f"{data.real.fold_change_col}_pred"].to_numpy()

            # weights: prefer (1 - q) on each side; else uniform
            w_true = None
            if getattr(data.real, "fdr_col", None) in df.columns:
                w_true = np.clip(1.0 - df[data.real.fdr_col].to_numpy(), 0.0, 1.0)
            else:
                w_true = np.ones(len(l_true), dtype=float)

            w_pred = None
            q_pred_col = getattr(data.pred, "fdr_col", None)
            if q_pred_col and f"{q_pred_col}_pred" in df.columns:
                w_pred = np.clip(1.0 - df[f"{q_pred_col}_pred"].to_numpy(), 0.0, 1.0)
            else:
                w_pred = np.ones(len(l_pred), dtype=float)

            w = w_true * w_pred
            if not np.any(w > 0):
                correlations[pert] = float("nan")
                continue

            # weighted Spearman (rank → weighted corr)
            r_true = np.argsort(np.argsort(l_true))
            r_pred = np.argsort(np.argsort(l_pred))
            rtx = r_true - np.average(r_true, weights=w)
            rty = r_pred - np.average(r_pred, weights=w)
            num = np.sum(w * rtx * rty)
            den = np.sqrt(np.sum(w * rtx**2) * np.sum(w * rty**2))
            rho_w = float(num / den) if den > 0 else float("nan")

            # overlap/precision factor (if provided); else fallback to simple Jaccard on FDR
            scale = 1.0
            if self.use_overlap:
                if pert in overlap:
                    scale = float(overlap[pert])
                else:
                    # quick fallback Jaccard based on FDR if both sides have it
                    if (getattr(data.real, "fdr_col", None) in df.columns) and q_pred_col and f"{q_pred_col}_pred" in df.columns:
                        S = df[data.real.fdr_col].to_numpy() < self.fdr_threshold
                        S_hat = df[f"{q_pred_col}_pred"].to_numpy() < self.fdr_threshold
                        u = (S | S_hat).sum()
                        scale = (S & S_hat).sum() / u if u > 0 else 0.0

            correlations[pert] = rho_w if np.isnan(rho_w) else rho_w * (scale ** self.gamma)

        return correlations

class DESpearmanSignificant:
    """Compute Spearman correlation on number of significant DE genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> float:
        """Compute correlation between number of significant genes in real and predicted DE."""

        filt_real = (
            data.real.filter_to_significant(fdr_threshold=self.fdr_threshold)
            .group_by(data.real.target_col)
            .len()
        )
        filt_pred = (
            data.pred.filter_to_significant(fdr_threshold=self.fdr_threshold)
            .group_by(data.pred.target_col)
            .len()
        )

        merged = filt_real.join(
            filt_pred,
            left_on=data.real.target_col,
            right_on=data.pred.target_col,
            suffix="_pred",
            how="left",
            coalesce=True,
        ).fill_null(0)

        # No significant genes in either real or predicted DE. Set to 1.0 since perfect
        # agreement but will fail spearman test
        if merged.shape[0] == 0:
            return 1.0

        return float(
            merged.select(
                pl.corr(
                    pl.col("len"),
                    pl.col("len_pred"),
                    method="spearman",
                ).alias("spearman_corr_nsig")
            )
            .to_numpy()
            .flatten()[0]
        )


class DEDirectionMatch:
    """Compute agreement in direction of DE gene changes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> dict[str, float]:
        """Compute directional agreement between real and predicted DE genes."""
        matches = {}

        merged = data.real.filter_to_significant(fdr_threshold=0.05).join(
            data.pred.data,
            on=[data.real.target_col, data.real.feature_col],
            suffix="_pred",
            how="inner",
        )
        for row in (
            merged.with_columns(
                direction_match=pl.col(data.real.log2_fold_change_col).sign()
                == pl.col(f"{data.real.log2_fold_change_col}_pred").sign()
            )
            .group_by(
                data.real.target_col,
            )
            .agg(pl.mean("direction_match"))
            .iter_rows()
        ):
            matches.update({row[0]: row[1]})
        return matches


class DESpearmanLFC:
    """Compute Spearman correlation on log fold changes of significant genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> dict[str, float]:
        """Compute correlation between log fold changes of significant genes."""
        correlations = {}

        merged = data.real.filter_to_significant(fdr_threshold=self.fdr_threshold).join(
            data.pred.data,
            on=[data.real.target_col, data.real.feature_col],
            suffix="_pred",
            how="inner",
        )

        for row in (
            merged.group_by(
                data.real.target_col,
            )
            .agg(
                pl.corr(
                    pl.col(data.real.fold_change_col),
                    pl.col(f"{data.real.fold_change_col}_pred"),
                    method="spearman",
                ).alias("spearman_corr"),
            )
            .iter_rows()
        ):
            correlations.update({row[0]: row[1]})

        return correlations


class DESigGenesRecall:
    """Compute recall of significant genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> dict[str, float]:
        """Compute recall of significant genes between real and predicted DE."""

        filt_real = data.real.filter_to_significant(fdr_threshold=self.fdr_threshold)
        filt_pred = data.pred.filter_to_significant(fdr_threshold=self.fdr_threshold)

        recall_frame = (
            filt_real.join(
                filt_pred,
                on=[data.real.target_col, data.real.feature_col],
                how="inner",
                coalesce=True,
            )
            .group_by(data.real.target_col)
            .len()
            .join(
                filt_real.group_by(data.real.target_col).len(),
                on=data.real.target_col,
                how="full",
                suffix="_expected",
                coalesce=True,
            )
            .fill_null(0)
            .with_columns(recall=pl.col("len") / pl.col("len_expected"))
            .select([data.real.target_col, "recall"])
        )

        return {row[0]: row[1] for row in recall_frame.iter_rows()}


class DENsigCounts:
    """Compute counts of significant genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> dict[str, dict[str, int]]:
        """Compute counts of significant genes in real and predicted DE."""
        counts = {}

        for pert in data.iter_perturbations():
            real_sig = data.real.get_significant_genes(pert, self.fdr_threshold)
            pred_sig = data.pred.get_significant_genes(pert, self.fdr_threshold)

            counts[pert] = {
                "real": int(real_sig.size),
                "pred": int(pred_sig.size),
            }

        return counts


def compute_pr_auc(data: DEComparison) -> dict[str, float]:
    """Compute precision-recall AUC per perturbation for significant recovery."""
    return compute_generic_auc(data, method="pr")


def compute_roc_auc(data: DEComparison) -> dict[str, float]:
    """Compute ROC AUC per perturbation for significant recovery."""
    return compute_generic_auc(data, method="roc")


def compute_generic_auc(
    data: DEComparison,
    method: Literal["pr", "roc"] = "pr",
) -> dict[str, float]:
    """Compute AUC values for significant recovery per perturbation."""

    target_col = data.real.target_col
    feature_col = data.real.feature_col
    real_fdr_col = data.real.fdr_col
    pred_fdr_col = data.pred.fdr_col

    labeled_real = data.real.data.with_columns(
        (pl.col(real_fdr_col) < 0.05).cast(pl.Float32).alias("label")
    ).select([target_col, feature_col, "label"])

    merged = (
        data.pred.data.select([target_col, feature_col, pred_fdr_col])
        .join(
            labeled_real,
            on=[target_col, feature_col],
            how="inner",
            coalesce=True,
        )
        .drop_nulls(["label"])
        .with_columns((-pl.col(pred_fdr_col).replace(0, 1e-10).log10()).alias("nlp"))
        .drop_nulls(["nlp"])
    )

    results: dict[str, float] = {}
    for pert in data.iter_perturbations():
        pert_data = merged.filter(pl.col(target_col) == pert)
        if pert_data.shape[0] == 0:
            results[pert] = float("nan")
            continue

        labels = pert_data["label"].to_numpy()
        scores = pert_data["nlp"].to_numpy()

        if not (0 < labels.sum() < len(labels)):
            results[pert] = float("nan")
            continue

        match method:
            case "pr":
                precision, recall, _ = precision_recall_curve(labels, scores)
                results[pert] = float(auc(recall, precision))
            case "roc":
                fpr, tpr, _ = roc_curve(labels, scores)
                results[pert] = float(auc(fpr, tpr))
            case _:
                raise ValueError(f"Invalid AUC method: {method}")

    return results
