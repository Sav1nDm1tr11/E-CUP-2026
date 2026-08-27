"""Losses for CC-OR-Net."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cascade_loss(logit1: torch.Tensor, logit2: torch.Tensor, bucket_true: torch.Tensor):
    """BCE cascade loss for the two chain-rule classifiers of CascadeHead.

    loss1 -- "y > 0?" -- over the whole batch.
    loss2 -- "y > tau2 | y > 0?" -- only over bucket_true > 0 rows. The
    conditional target is undefined for bucket_true == 0 (not "false"), so
    a fabricated label would be wrong -- those rows are masked out of loss2
    entirely instead. If the batch has no bucket_true > 0 rows at all, loss2
    is 0.0 (via the clamp_min(1.0) denominator), never NaN.
    """
    logit1 = logit1.squeeze(-1)
    logit2 = logit2.squeeze(-1)
    bucket_true = bucket_true.float()

    loss1 = F.binary_cross_entropy_with_logits(logit1, (bucket_true > 0).float())

    mask = (bucket_true > 0).float()
    raw2 = F.binary_cross_entropy_with_logits(
        logit2, (bucket_true > 1).float(), reduction="none"
    )
    loss2 = (raw2 * mask).sum() / mask.sum().clamp_min(1.0)

    total = loss1 + loss2
    return total, {"loss1": loss1, "loss2": loss2}


def regression_loss(
    v_norm: torch.Tensor,
    y: torch.Tensor,
    bucket_true: torch.Tensor,
    denorm_stats: dict,
):
    """SmoothL1 loss on v_norm against per-bucket denorm targets.

    v_target = clip((log1p(y) - c_b) / r_b, -1, 1) is computed with the
    TRUE bucket (teacher forcing), not the predicted one -- otherwise, early
    in training when the cascade is still noisy, the regression target
    would pick up the wrong (r_b, c_b) and become a noisy/incorrect signal.

    Only bucket_true > 0 rows count -- regression for bucket 0 is undefined
    (final prediction there is always 0 by construction), so those rows are
    masked out, exactly like loss2 in cascade_loss. If the batch has no
    positive-bucket rows, loss is 0.0 (via clamp_min(1.0)), never NaN.
    """
    bucket_true = bucket_true.long()
    log_y = torch.log1p(y.float())

    r = torch.ones_like(log_y)
    c = torch.zeros_like(log_y)
    for bucket, (r_b, c_b) in denorm_stats.items():
        bucket_mask = bucket_true == bucket
        r = torch.where(bucket_mask, torch.full_like(r, float(r_b)), r)
        c = torch.where(bucket_mask, torch.full_like(c, float(c_b)), c)

    v_target = ((log_y - c) / r).clamp(-1.0, 1.0)

    mask = (bucket_true > 0).float()
    raw = F.smooth_l1_loss(v_norm, v_target, reduction="none")
    loss = (raw * mask).sum() / mask.sum().clamp_min(1.0)

    return loss, {"v_target": v_target}


def total_loss(
    outputs: dict,
    y: torch.Tensor,
    bucket_true: torch.Tensor,
    denorm_stats: dict,
    whale_module,
    w_cascade: float = 1.0,
    w_reg: float = 1.0,
    w_whale: float = 0.1,
):
    """Combines cascade_loss + regression_loss + w_whale * whale_auxiliary_loss.

    ``outputs`` is CCORNet's forward() dict (logit1, logit2, bucket_probs,
    v_norm, g). ``whale_module`` is a WhaleAugmentation instance -- called
    explicitly here (training loop), never inside CCORNet.forward().
    """
    cascade_total, cascade_components = cascade_loss(
        outputs["logit1"], outputs["logit2"], bucket_true
    )
    reg_loss, _ = regression_loss(outputs["v_norm"], y, bucket_true, denorm_stats)
    whale_loss = whale_module.whale_auxiliary_loss(outputs["g"], bucket_true)

    total = w_cascade * cascade_total + w_reg * reg_loss + w_whale * whale_loss

    components = {
        "loss1": cascade_components["loss1"],
        "loss2": cascade_components["loss2"],
        "loss_reg": reg_loss,
        "loss_whale": whale_loss,
        "total": total,
    }
    return total, components
