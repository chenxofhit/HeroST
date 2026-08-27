import torch

from heroST.data.dataset import ExpressionNormalizer, SlideGraphDataset
from heroST.data.synthetic import generate_synthetic_cohort
from heroST.models.hero_st import HeroSTConfig, HeroSTModel


def _batch():
    slides = generate_synthetic_cohort(
        n_slides=3,
        grid_size=5,
        n_domains=3,
        n_genes=8,
        n_marker_genes=4,
        n_immune_genes=2,
        patch_size=8,
        seed=7,
    )
    dataset = SlideGraphDataset(slides, ExpressionNormalizer().fit(slides), knn_k=4)
    return dataset[0]


def test_core_forward_is_image_only_at_inference():
    batch = _batch()
    model = HeroSTModel(
        HeroSTConfig(
            n_genes=batch.expression.shape[1],
            embed_dim=16,
            n_message_passing_layers=2,
            m2_nonneg_weights=True,
            m2_monotone_gate=True,
            m2_boundary_penalty_mode="dst",
        )
    )

    training_output = model(
        batch.patches,
        batch.coords,
        batch.spatial_edge_index,
        batch.spatial_edge_dist,
        expression_true=batch.expression,
    )
    inference_output = model(
        batch.patches,
        batch.coords,
        batch.spatial_edge_index,
        batch.spatial_edge_dist,
    )

    assert training_output.m3.mu_teacher is not None
    assert inference_output.m3.mu_teacher is None
    assert inference_output.m3.mu.shape == batch.expression.shape


def test_constrained_propagation_backward_pass():
    batch = _batch()
    model = HeroSTModel(
        HeroSTConfig(
            n_genes=batch.expression.shape[1],
            embed_dim=16,
            m2_nonneg_weights=True,
            m2_monotone_gate=True,
            m2_boundary_penalty_mode="dst",
        )
    )
    output = model(
        batch.patches,
        batch.coords,
        batch.spatial_edge_index,
        batch.spatial_edge_dist,
        expression_true=batch.expression,
    )
    loss = torch.nn.functional.mse_loss(output.m3.mu, batch.expression)
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    assert all(value >= 0.0 for value in model.m2.weight_report().values())
