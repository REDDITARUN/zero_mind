from __future__ import annotations

from pathlib import Path
import torch

from src.models.unified_model import UnifiedArcModel
from src.training.inference import load_model_for_inference


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = UnifiedArcModel(num_colors=10, dim=64, depth=2, heads=4)
    ckpt = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model_cfg": {"num_colors": 10, "dim": 64, "depth": 2, "heads": 4},
            "state_dict": model.state_dict(),
        },
        ckpt,
    )
    loaded = load_model_for_inference(str(ckpt), device="cpu")
    assert isinstance(loaded, UnifiedArcModel)
