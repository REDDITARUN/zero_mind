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
from src.models.decoder_refine import DraftCorrectDecoder, DecodeOutput
from src.models.symbolic_primitives import fit_symbolic_rule


@dataclass
class UnifiedForwardOutput:
    test_logits: torch.Tensor
    draft_logits: torch.Tensor
    aux_logits: torch.Tensor
    router: RouterOutput
    reasoner: ReasonerOutput
    decoder: DecodeOutput
    test_shape: torch.Size
    symbolic_confidence: torch.Tensor
    symbolic_rule: str


class UnifiedArcModel(nn.Module):
    """v4: HRM + Draft-Correct Decoder (System 1 + System 2).

    Architecture:
      Encoder (RoPE Attention) → Cross-Attention Rule Extractor →
        Content Reasoner (HRM: H/L cycles with 1-step grad trick) →
        Draft-Correct Decoder:
          Stage 1 — Draft: fast MLP refinement (strong gradients)
          Stage 2 — Correct: re-inject draft, self-attend to fix errors
      + Auxiliary direct readout (gradient shortcut for encoder)
      + Symbolic blend at inference

    Key changes from v3:
      - Removed shape reasoner/predictor (wasted compute for same-shape)
      - DraftCorrectDecoder replaces CrossAttn ContentDecoder
        (direct token refinement >> generating from scratch canvas)
      - Two learning signals: draft loss + corrector loss
      - Simpler, faster, stronger gradients
    """

    def __init__(
        self,
        num_colors: int = 10,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        h_layers: int = 3,
        l_layers: int = 3,
        h_cycles: int = 3,
        l_cycles: int = 2,
        draft_steps: int = 3,
        correct_steps: int = 3,
        use_mamba: bool = False,
        symbolic_train_cap: float = 0.0,
        symbolic_infer_cap: float = 1.0,
        # Legacy compat — silently ignored
        dino_hidden: int = 0,
        dino_out: int = 0,
        pred_depth: int | None = None,
        reasoner_max_steps: int | None = None,
        reasoner_depth: int | None = None,
        shape_reasoner_depth: int | None = None,
        shape_h_layers: int | None = None,
        shape_l_layers: int | None = None,
        shape_h_cycles: int | None = None,
        shape_l_cycles: int | None = None,
        decoder_refine_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = IJepaEncoder2DRoPE(
            num_colors=num_colors + 1, dim=dim, depth=depth, heads=heads, use_mamba=use_mamba,
        )

        self.rule = CrossAttentionRuleExtractor(dim=dim, heads=heads)

        self.router = NoveltyRouter(dim=dim)

        _h = h_layers if reasoner_depth is None else reasoner_depth
        _l = l_layers if reasoner_depth is None else reasoner_depth
        self.content_reasoner = TrmHrmReasoner(
            dim=dim, h_layers=_h, l_layers=_l, h_cycles=h_cycles, l_cycles=l_cycles, heads=heads,
        )

        _ds = draft_steps if decoder_refine_steps is None else max(1, decoder_refine_steps // 2)
        _cs = correct_steps if decoder_refine_steps is None else max(1, decoder_refine_steps - _ds)
        self.decoder = DraftCorrectDecoder(
            dim=dim, num_colors=num_colors, draft_steps=_ds, correct_steps=_cs, heads=heads,
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

        content_out = self.content_reasoner(rule_out.test_conditioned_latent, h=h_in, w=w_in)

        decode_out = self.decoder(content_out.latent, h=h_in, w=w_in)

        neural_logits = decode_out.logits.reshape(1, h_in, w_in, -1).permute(0, 3, 1, 2)
        draft_logits_2d = decode_out.draft_logits.reshape(1, h_in, w_in, -1).permute(0, 3, 1, 2)

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
            draft_logits=draft_logits_2d,
            aux_logits=aux_logits,
            router=router_out,
            reasoner=content_out,
            decoder=decode_out,
            test_shape=test_input.shape,
            symbolic_confidence=torch.tensor(sym_w, device=test_input.device),
            symbolic_rule=sym.rule_name,
        )
