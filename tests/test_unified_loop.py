from __future__ import annotations

import torch

from src.data.arc_simulator import ArcSimulator
from src.models.unified_model import UnifiedArcModel
from src.training.unified_loop import TrainConfig, UnifiedTrainer, run_inference, run_inference_ttt


def test_single_train_step_runs() -> None:
    device = torch.device("cpu")
    sim = ArcSimulator(device=device, seed=1)
    ep = sim.sample_episode()

    model = UnifiedArcModel(num_colors=10, dim=64, depth=2, heads=4)
    trainer = UnifiedTrainer(model=model, cfg=TrainConfig(steps=10, device="cpu", grad_accum_steps=1))
    metrics = trainer.train_step(ep, step=1)

    assert "total" in metrics
    assert "content" in metrics
    assert "aux" in metrics
    assert "pixel_acc" in metrics
    assert 0.0 <= metrics["alpha_sft"] <= 1.0
    assert 0.0 <= metrics["alpha_rl"] <= 1.0


def test_inference_shape_matches_target() -> None:
    device = torch.device("cpu")
    sim = ArcSimulator(device=device, seed=2)
    ep = sim.sample_episode()

    model = UnifiedArcModel(num_colors=10, dim=64, depth=2, heads=4)
    pred = run_inference(model, ep)
    assert pred.shape == ep.test_output.shape


def test_ttt_inference_runs() -> None:
    device = torch.device("cpu")
    sim = ArcSimulator(device=device, seed=3)
    ep = sim.sample_episode()

    model = UnifiedArcModel(num_colors=10, dim=64, depth=2, heads=4)
    pred = run_inference_ttt(model, ep, ttt_steps=2, ttt_lr=1e-4)
    assert pred.shape == ep.test_output.shape

    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    run_inference_ttt(model, ep, ttt_steps=2, ttt_lr=1e-4)
    for k in state_before:
        assert torch.equal(state_before[k], model.state_dict()[k]), f"TTT leaked weights for {k}"
