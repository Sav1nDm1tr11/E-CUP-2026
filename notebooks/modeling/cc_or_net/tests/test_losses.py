import numpy as np
import torch

from src.inference import assemble_prediction
from src.losses import cascade_loss, regression_loss, total_loss
from src.model import WhaleAugmentation


def test_cascade_loss_finite_and_sums_components():
    torch.manual_seed(0)
    logit1 = torch.randn(16, 1)
    logit2 = torch.randn(16, 1)
    bucket_true = torch.randint(0, 3, (16,))

    total, components = cascade_loss(logit1, logit2, bucket_true)

    assert torch.isfinite(total)
    assert torch.isfinite(components["loss1"])
    assert torch.isfinite(components["loss2"])
    assert torch.allclose(total, components["loss1"] + components["loss2"])


def test_cascade_loss_no_positive_bucket_gives_zero_loss2_not_nan():
    logit1 = torch.randn(8, 1)
    logit2 = torch.randn(8, 1)
    bucket_true = torch.zeros(8, dtype=torch.int64)

    total, components = cascade_loss(logit1, logit2, bucket_true)

    assert torch.isfinite(total)
    assert components["loss2"].item() == 0.0


def test_cascade_loss_loss2_ignores_bucket_zero_rows_value():
    bucket_true = torch.tensor([0, 0, 1, 2, 1, 0])
    logit1 = torch.randn(6, 1)
    logit2_a = torch.randn(6, 1)
    logit2_b = logit2_a.clone()

    mask0 = bucket_true == 0
    # "испорченный" logit2 только у bucket_true==0 строк -- условная
    # вероятность "y>tau2 | y>0" для них не определена, эти значения не
    # должны влиять на loss2.
    logit2_b[mask0] = 1000.0

    _, components_a = cascade_loss(logit1, logit2_a, bucket_true)
    _, components_b = cascade_loss(logit1, logit2_b, bucket_true)

    assert torch.isfinite(components_b["loss2"])
    assert torch.allclose(components_a["loss2"], components_b["loss2"])


def test_cascade_loss_loss2_gradient_ignores_bucket_zero_rows():
    torch.manual_seed(2)
    bucket_true = torch.tensor([0, 0, 1, 2, 1, 0])
    logit1 = torch.randn(6, 1)
    logit2 = torch.randn(6, 1, requires_grad=True)

    _, components = cascade_loss(logit1, logit2, bucket_true)
    components["loss2"].backward()

    mask0 = bucket_true == 0
    assert torch.allclose(logit2.grad[mask0], torch.zeros_like(logit2.grad[mask0]))
    assert not torch.allclose(logit2.grad[~mask0], torch.zeros_like(logit2.grad[~mask0]))


DENORM_STATS = {1: (1.0, 3.0), 2: (1.0, 5.0)}


def test_regression_loss_no_positive_bucket_gives_zero_not_nan():
    v_norm = torch.randn(8)
    y = torch.zeros(8)
    bucket_true = torch.zeros(8, dtype=torch.int64)

    loss, _ = regression_loss(v_norm, y, bucket_true, DENORM_STATS)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_regression_loss_ignores_bucket_zero_rows_value():
    bucket_true = torch.tensor([0, 1, 2, 0, 1, 0])
    y = torch.tensor([0.0, 20.0, 200.0, 0.0, 30.0, 0.0])
    v_norm_a = torch.randn(6)
    v_norm_b = v_norm_a.clone()

    mask0 = bucket_true == 0
    # "испорченный" v_norm только у bucket_true==0 строк -- регрессия для
    # бакета 0 не определена, эти значения не должны влиять на loss.
    v_norm_b[mask0] = 999.0

    loss_a, _ = regression_loss(v_norm_a, y, bucket_true, DENORM_STATS)
    loss_b, _ = regression_loss(v_norm_b, y, bucket_true, DENORM_STATS)

    assert torch.isfinite(loss_b)
    assert torch.allclose(loss_a, loss_b)


def test_regression_loss_loss2_gradient_ignores_bucket_zero_rows():
    torch.manual_seed(5)
    bucket_true = torch.tensor([0, 1, 2, 0, 1, 0])
    y = torch.tensor([0.0, 20.0, 200.0, 0.0, 30.0, 0.0])
    v_norm = torch.randn(6, requires_grad=True)

    loss, _ = regression_loss(v_norm, y, bucket_true, DENORM_STATS)
    loss.backward()

    mask0 = bucket_true == 0
    assert torch.allclose(v_norm.grad[mask0], torch.zeros_like(v_norm.grad[mask0]))
    assert not torch.allclose(v_norm.grad[~mask0], torch.zeros_like(v_norm.grad[~mask0]))


def test_regression_loss_round_trip_with_assemble_prediction():
    # y внутри диапазона [P10, P90] бакета -- гарантируем, что clip к
    # [-1, 1] в v_target не сработает, и round-trip
    # (y -> v_target -> assemble_prediction) вернёт исходный y.
    r_b, c_b = 1.0, 4.0
    v_target_expected = 0.4
    y = float(np.expm1(c_b + v_target_expected * r_b))

    denorm_stats = {1: (r_b, c_b), 2: (2.0, 6.0)}
    bucket_true = torch.tensor([1])
    v_norm = torch.tensor([v_target_expected])
    y_tensor = torch.tensor([y])

    loss, extra = regression_loss(v_norm, y_tensor, bucket_true, denorm_stats)
    torch.testing.assert_close(extra["v_target"], v_norm, atol=1e-5, rtol=0)
    assert loss.item() < 1e-4

    bucket_probs = np.array([[0.0, 1.0, 0.0]])  # argmax -> bucket 1
    pred = assemble_prediction(bucket_probs, v_norm.numpy(), denorm_stats)

    np.testing.assert_allclose(pred, [y], rtol=1e-4)


def test_total_loss_combines_components_with_weights():
    torch.manual_seed(7)
    batch = 12
    outputs = {
        "logit1": torch.randn(batch, 1),
        "logit2": torch.randn(batch, 1),
        "v_norm": torch.randn(batch),
        "g": torch.randn(batch, 128),
    }
    y = torch.rand(batch) * 100
    bucket_true = torch.randint(0, 3, (batch,))
    whale = WhaleAugmentation(h_size=128)

    total, components = total_loss(
        outputs, y, bucket_true, DENORM_STATS, whale,
        w_cascade=1.0, w_reg=1.0, w_whale=0.1,
    )

    assert set(components.keys()) == {"loss1", "loss2", "loss_reg", "loss_whale", "total"}
    assert torch.isfinite(total)

    # whale_auxiliary_loss injects random noise, so it can't be recomputed
    # bit-for-bit outside total_loss -- instead check total is consistent
    # with the OTHER (deterministic) components it actually returned, plus
    # its own returned loss_whale.
    cascade_total, cascade_components = cascade_loss(outputs["logit1"], outputs["logit2"], bucket_true)
    reg_loss, _ = regression_loss(outputs["v_norm"], y, bucket_true, DENORM_STATS)
    expected_total = cascade_total + reg_loss + 0.1 * components["loss_whale"]

    torch.testing.assert_close(total, expected_total)
    torch.testing.assert_close(components["loss1"], cascade_components["loss1"])
    torch.testing.assert_close(components["loss2"], cascade_components["loss2"])
    torch.testing.assert_close(components["loss_reg"], reg_loss)
    assert torch.isfinite(components["loss_whale"])
    assert components["loss_whale"].item() >= 0.0


def test_total_loss_backward_reaches_whale_and_cascade_params():
    torch.manual_seed(8)
    batch = 12
    h = torch.randn(batch, 128, requires_grad=True)

    whale = WhaleAugmentation(h_size=128)
    outputs = {
        "logit1": torch.randn(batch, 1, requires_grad=True),
        "logit2": torch.randn(batch, 1, requires_grad=True),
        "v_norm": torch.randn(batch, requires_grad=True),
        "g": h,
    }
    y = torch.rand(batch) * 100
    bucket_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    total, _ = total_loss(outputs, y, bucket_true, DENORM_STATS, whale, w_whale=0.1)
    total.backward()

    assert whale.attn.weight.grad is not None
    assert whale.aux_head.weight.grad is not None
    assert torch.isfinite(whale.attn.weight.grad).all()
