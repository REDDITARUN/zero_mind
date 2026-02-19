from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from src.models.ijepa_rope import IJepaEncoder2DRoPE
from src.models.cross_attention_rule import CrossAttentionRuleExtractor
from src.models.router import NoveltyRouter, RouterOutput
from src.models.reasoner_trm_hrm import TrmHrmReasoner, ReasonerOutput
from src.models.decoder_refine import ContentDecoder, DecodeOutput
from src.models.symbolic_primitives import fit_symbolic_rule


class ShapePredictor(nn.Module):
    """Predicts output grid dimensions from pooled shape-reasoner latent.

    Classifies H and W independently over [1, max_size].
    """

    def __init__(self, dim: int = 256, max_size: int = 30) -> None:
        super().__init__()
        self.max_size = max_size
        self.norm = nn.LayerNorm(dim)
        self.h_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, max_size))
        self.w_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, max_size))

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.norm(latent.mean(dim=1))
        return self.h_head(pooled), self.w_head(pooled)


@dataclass
class UnifiedForwardOutput:
    test_logits: torch.Tensor
    aux_logits: torch.Tensor
    shape_h_logits: torch.Tensor
    shape_w_logits: torch.Tensor
    router: RouterOutput
    reasoner: ReasonerOutput          # content reasoner (backward compat)
    shape_reasoner: ReasonerOutput
    decoder: DecodeOutput
    test_shape: torch.Size
    target_shape: tuple[int, int]
    symbolic_confidence: torch.Tensor
    symbolic_rule: str


class UnifiedArcModel(nn.Module):
    """v3: Dual-Head HRM + Cross-Attention Content Decoder.

    Architecture:
      Encoder (RoPE Attention) → Cross-Attention Rule Extractor →
        ┌── Content Reasoner (full HRM: deep H/L cycles) → Content Decoder
        └── Shape Reasoner (lightweight HRM) → Shape Predictor (H, W)
      + Auxiliary direct readout (gradient shortcut for encoder)
      + Symbolic blend

    Key changes from v2:
      - No DINO (removed target encoder + projection heads)
      - Dual parallel HRM loops (brain-inspired "what" + "where" pathways)
      - Cross-attention content decoder (canvas + diffusion-style refinement)
      - Shape predictor (prepares for variable-size output support)
    """

    def __init__(
        self,
        num_colors: int = 10,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        # Content reasoner (full power)
        reasoner_depth: int = 3,
        h_cycles: int = 3,
        l_cycles: int = 2,
        # Shape reasoner (lightweight parallel loop)
        shape_reasoner_depth: int = 1,
        shape_h_cycles: int = 1,
        shape_l_cycles: int = 1,
        # Content decoder
        decoder_refine_steps: int = 6,
        # Encoder options
        use_mamba: bool = False,
        # Symbolic
        symbolic_train_cap: float = 0.45,
        symbolic_infer_cap: float = 1.0,
        # Legacy compat — silently ignored
        dino_hidden: int = 0,
        dino_out: int = 0,
        pred_depth: int | None = None,
        reasoner_max_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = IJepaEncoder2DRoPE(
            num_colors=num_colors + 1, dim=dim, depth=depth, heads=heads, use_mamba=use_mamba,
        )

        self.rule = CrossAttentionRuleExtractor(dim=dim, heads=heads)

        self.router = NoveltyRouter(dim=dim)

        self.content_reasoner = TrmHrmReasoner(
            dim=dim, depth=reasoner_depth, h_cycles=h_cycles, l_cycles=l_cycles, heads=heads,
        )
        self.shape_reasoner = TrmHrmReasoner(
            dim=dim, depth=shape_reasoner_depth,
            h_cycles=shape_h_cycles, l_cycles=shape_l_cycles,
            heads=heads, max_experts=2, init_active_experts=1,
        )

        self.shape_predictor = ShapePredictor(dim=dim, max_size=30)

        self.decoder = ContentDecoder(
            dim=dim, num_colors=num_colors, refine_steps=decoder_refine_steps, heads=heads,
        )

        self.aux_readout = nn.Linear(dim, num_colors)

        self.num_colors = num_colors
        self.symbolic_train_cap = symbolic_train_cap
        self.symbolic_infer_cap = symbolic_infer_cap

    @torch.no_grad()
    def adapt_structure(
        self,
        reward: float,
        novelty: float,
        uncertainty: float,
        hard_cap: int | None = None,
        target_cap: int | None = None,
        grow_threshold: float = 0.55,
        prune_threshold: float = 0.15,
        ema_momentum: float = 0.9,
        prune_cooldown_steps: int = 0,
    ) -> tuple[bool, bool]:
        return self.content_reasoner.adapt_structure(
            reward=reward,
            novelty=novelty,
            uncertainty=uncertainty,
            hard_cap=hard_cap,
            target_cap=target_cap,
            grow_threshold=grow_threshold,
            prune_threshold=prune_threshold,
            ema_momentum=ema_momentum,
            prune_cooldown_steps=prune_cooldown_steps,
        )

    def encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        return self.encoder(grid.unsqueeze(0))

    def forward(
        self,
        train_inputs: List[torch.Tensor],
        train_outputs: List[torch.Tensor],
        test_input: torch.Tensor,
    ) -> UnifiedForwardOutput:
        train_in_latents, train_out_latents = [], []

        for g in train_inputs:
            train_in_latents.append(self.encode_grid(g))
        for g in train_outputs:
            train_out_latents.append(self.encode_grid(g))

        test_latent = self.encode_grid(test_input)

        rule_out = self.rule(
            test_latent=test_latent,
            train_input_latents=train_in_latents,
            train_output_latents=train_out_latents,
        )

        router_out = self.router(rule_out.test_conditioned_latent)

        h_in, w_in = test_input.shape

        # Dual HRM: two parallel reasoning pathways
        content_out = self.content_reasoner(rule_out.test_conditioned_latent, h=h_in, w=w_in)
        shape_out = self.shape_reasoner(rule_out.test_conditioned_latent, h=h_in, w=w_in)

        shape_h_logits, shape_w_logits = self.shape_predictor(shape_out.latent)

        h_out, w_out = h_in, w_in
        target_shape = (h_out, w_out)

        decode_out = self.decoder(content_out.latent, h_out, w_out)

        neural_logits = decode_out.logits.reshape(1, h_out, w_out, -1).permute(0, 3, 1, 2)

        aux_logits_raw = self.aux_readout(rule_out.test_conditioned_latent)
        aux_logits = aux_logits_raw.reshape(1, h_in, w_in, -1).permute(0, 3, 1, 2)

        sym = fit_symbolic_rule(train_inputs=train_inputs, train_outputs=train_outputs, test_input=test_input)
        sym_one_hot = F.one_hot(sym.grid.clamp(0, self.num_colors - 1), num_classes=self.num_colors).float()
        sym_logits = sym_one_hot.permute(2, 0, 1).unsqueeze(0) * 6.0
        sym_w = float(max(0.0, min(1.0, sym.confidence)))
        cap = self.symbolic_train_cap if self.training else self.symbolic_infer_cap
        sym_w = float(min(sym_w, max(0.0, min(1.0, cap))))
        test_logits = (1.0 - sym_w) * neural_logits + sym_w * sym_logits

        return UnifiedForwardOutput(
            test_logits=test_logits,
            aux_logits=aux_logits,
            shape_h_logits=shape_h_logits,
            shape_w_logits=shape_w_logits,
            router=router_out,
            reasoner=content_out,
            shape_reasoner=shape_out,
            decoder=decode_out,
            test_shape=test_input.shape,
            target_shape=target_shape,
            symbolic_confidence=torch.tensor(sym_w, device=test_input.device),
            symbolic_rule=sym.rule_name,
        )
