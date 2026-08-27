import ast
import inspect
import textwrap

import numpy as np
import pytest
import torch

from src.inference import assemble_prediction, predict_submission
from src.model import (
    CascadeHead,
    CCORNet,
    FeatureAlignment,
    ResidualBlock,
    ResidualRegressionHead,
    WhaleAugmentation,
)


def test_cascade_head_output_shapes():
    head = CascadeHead(h_size=128)
    h = torch.randn(32, 128)

    logit1, logit2, bucket_probs = head(h)

    assert logit1.shape == (32, 1)
    assert logit2.shape == (32, 1)
    assert bucket_probs.shape == (32, 3)


def test_cascade_head_gate2_sees_h_and_logit1():
    # gate2 видит concat(h, logit1) -> in_features == h_size + 1.
    head = CascadeHead(h_size=128)
    assert head.gate2.in_features == 129


def test_cascade_head_bucket_probs_sum_to_one():
    torch.manual_seed(0)
    head = CascadeHead(h_size=128)
    h = torch.randn(64, 128)

    _, _, bucket_probs = head(h)

    torch.testing.assert_close(
        bucket_probs.sum(dim=-1), torch.ones(64), atol=1e-5, rtol=0
    )


def test_cascade_head_bucket_probs_match_chain_rule():
    torch.manual_seed(1)
    head = CascadeHead(h_size=128)
    h = torch.randn(16, 128)

    logit1, logit2, bucket_probs = head(h)
    p1 = torch.sigmoid(logit1).squeeze(-1)
    p2 = torch.sigmoid(logit2).squeeze(-1)

    expected = torch.stack([1 - p1, p1 * (1 - p2), p1 * p2], dim=-1)
    torch.testing.assert_close(bucket_probs, expected)


def test_feature_alignment_output_shape():
    align = FeatureAlignment(h_size=128, bucket_embed_size=32)
    h = torch.randn(16, 128)
    bucket_probs = torch.softmax(torch.randn(16, 3), dim=-1)

    g = align(h, bucket_probs)

    assert g.shape == (16, 128)
    assert torch.isfinite(g).all()


def test_feature_alignment_passes_gradient_back_into_cascade_head():
    # bucket_probs участвует в gate_input, поэтому градиент от g должен
    # доходить до параметров CascadeHead, а не отсекаться на FeatureAlignment.
    torch.manual_seed(3)
    cascade_head = CascadeHead(h_size=128)
    align = FeatureAlignment(h_size=128, bucket_embed_size=32)

    h = torch.randn(8, 128)
    _, _, bucket_probs = cascade_head(h)
    g = align(h, bucket_probs)
    g.sum().backward()

    assert cascade_head.gate1.weight.grad is not None
    assert cascade_head.gate2.weight.grad is not None
    assert torch.isfinite(cascade_head.gate1.weight.grad).all()
    assert torch.isfinite(cascade_head.gate2.weight.grad).all()


def test_residual_block_preserves_shape():
    block = ResidualBlock(size=128)
    x = torch.randn(10, 128)

    out = block(x)

    assert out.shape == (10, 128)
    assert torch.isfinite(out).all()


def test_residual_regression_head_output_shape_and_range():
    torch.manual_seed(4)
    head = ResidualRegressionHead(h_size=128)
    g = torch.randn(20, 128)

    v_norm = head(g)

    assert v_norm.shape == (20,)
    assert torch.isfinite(v_norm).all()
    assert bool((v_norm >= -1.0).all() and (v_norm <= 1.0).all())


def test_assemble_prediction_bucket_zero_is_always_zero_regardless_of_v_norm():
    bucket_probs = np.array([
        [0.9, 0.05, 0.05],  # b_pred = 0
        [0.1, 0.8, 0.1],  # b_pred = 1
        [0.1, 0.1, 0.8],  # b_pred = 2
    ])
    denorm_stats = {1: (1.0, 3.0), 2: (1.0, 5.0)}

    v_norm_a = np.array([-1.0, 0.2, -0.3])
    v_norm_b = np.array([1.0, 0.2, -0.3])  # "испорченный" v_norm только у bucket-0 строки

    pred_a = assemble_prediction(bucket_probs, v_norm_a, denorm_stats)
    pred_b = assemble_prediction(bucket_probs, v_norm_b, denorm_stats)

    assert pred_a[0] == 0.0
    assert pred_b[0] == 0.0
    np.testing.assert_allclose(pred_a[1:], pred_b[1:])


def test_assemble_prediction_soft_matches_log_space_blend():
    # soft: log1p(pred) = sum_b P(b) * (v_norm * r_b + c_b), вклад b=0 равен 0
    bucket_probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    v_norm = np.array([0.4, -0.2])
    denorm_stats = {1: (1.0, 3.0), 2: (2.0, 5.0)}

    pred = assemble_prediction(bucket_probs, v_norm, denorm_stats, mode="soft")

    expected_log = (
        bucket_probs[:, 1] * (v_norm * 1.0 + 3.0)
        + bucket_probs[:, 2] * (v_norm * 2.0 + 5.0)
    )
    np.testing.assert_allclose(pred, np.expm1(expected_log))


def test_assemble_prediction_soft_equals_argmax_when_fully_confident():
    # при P(b)=1 на одном бакете soft и argmax обязаны совпасть
    bucket_probs = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    v_norm = np.array([0.3, -0.5])
    denorm_stats = {1: (1.0, 3.0), 2: (2.0, 5.0)}

    hard = assemble_prediction(bucket_probs, v_norm, denorm_stats, mode="argmax")
    soft = assemble_prediction(bucket_probs, v_norm, denorm_stats, mode="soft")

    np.testing.assert_allclose(hard, soft, rtol=1e-9)


def test_assemble_prediction_soft_is_zero_only_when_bucket0_certain():
    # P(b0)=1 -> log1p(pred)=0 -> pred=0; иначе soft НЕ обязан давать ровно 0
    denorm_stats = {1: (1.0, 3.0), 2: (2.0, 5.0)}
    certain_zero = assemble_prediction(
        np.array([[1.0, 0.0, 0.0]]), np.array([0.9]), denorm_stats, mode="soft"
    )
    assert certain_zero[0] == 0.0

    mixed = assemble_prediction(
        np.array([[0.9, 0.1, 0.0]]), np.array([0.0]), denorm_stats, mode="soft"
    )
    assert mixed[0] > 0.0


def test_assemble_prediction_rejects_unknown_mode():
    with pytest.raises(ValueError):
        assemble_prediction(
            np.array([[1.0, 0.0, 0.0]]), np.array([0.0]), {1: (1.0, 3.0)}, mode="bogus"
        )


def test_whale_auxiliary_loss_no_whale_rows_returns_zero_not_nan():
    whale = WhaleAugmentation(h_size=128)
    g = torch.randn(10, 128)
    bucket_true = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])

    loss = whale.whale_auxiliary_loss(g, bucket_true)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_whale_auxiliary_loss_positive_finite_and_backward_reaches_whale_params():
    torch.manual_seed(6)
    whale = WhaleAugmentation(h_size=128)
    g = torch.randn(10, 128, requires_grad=True)
    bucket_true = torch.tensor([0, 2, 1, 2, 0, 2, 1, 0, 2, 1])

    loss = whale.whale_auxiliary_loss(g, bucket_true)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()

    assert whale.attn.weight.grad is not None
    assert whale.aux_head.weight.grad is not None
    assert torch.isfinite(whale.attn.weight.grad).all()
    assert torch.isfinite(whale.aux_head.weight.grad).all()


def test_ccornet_forward_does_not_reference_whale_augmentation():
    # Структурная изоляция: CCORNet.__init__/forward() не должны ни
    # создавать, ни вызывать WhaleAugmentation -- это отдельный компонент,
    # участвующий только в training loop через явный вызов
    # whale_auxiliary_loss. Проверяем именно код __init__/forward (не
    # class-level docstring, который сам объясняет это исключение прозой и
    # закономерно упоминает "WhaleAugmentation").
    init_source = textwrap.dedent(inspect.getsource(CCORNet.__init__))
    forward_source = textwrap.dedent(inspect.getsource(CCORNet.forward))

    for source in (init_source, forward_source):
        tree = ast.parse(source)
        # убираем docstring метода (первый Expr/Constant в теле функции),
        # чтобы текстовая проверка не спотыкалась о комментарии в прозе.
        body = tree.body[0].body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        code_only = "\n".join(ast.unparse(node) for node in body)
        assert "whale" not in code_only.lower()

        names_used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs_used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert not any("whale" in name.lower() for name in names_used)
        assert not any("whale" in attr.lower() for attr in attrs_used)


def test_predict_submission_does_not_reference_whale_module():
    # whale_module_unused -- принят в сигнатуре только для полноты
    # чекпоинт-бандла, но по архитектуре WhaleAugmentation не участвует в
    # инференсе: тело функции не должно ни разу его использовать.
    source = textwrap.dedent(inspect.getsource(predict_submission))
    tree = ast.parse(source)
    func_def = tree.body[0]
    body = func_def.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]

    names_used = {
        node.id
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
    }
    assert "whale_module_unused" not in names_used
