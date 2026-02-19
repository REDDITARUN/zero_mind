from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from src.models.ijepa_rope import IJepaEncoder2DRoPE, IJepaPredictor
from src.models.cross_attention_rule import CrossAttentionRuleExtractor
from src.models.router import NoveltyRouter, RouterOutput
from src.models.reasoner_trm_hrm import TrmHrmReasoner, ReasonerOutput
from src.models.decoder_refine import IterativeRefinementDecoder, DecodeOutput
from src.models.symbolic_primitives import fit_symbolic_rule


@dataclass
class UnifiedForwardOutput:
    test_logits: torch.Tensor
    ijepa_loss: torch.Tensor
    router: RouterOutput
    reasoner: ReasonerOutput
    decoder: DecodeOutput
    test_shape: torch.Size
    symbolic_confidence: torch.Tensor
    symbolic_rule: str


class UnifiedArcModel(nn.Module):
    def __init__(
        self,
        num_colors: int = 10,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        reasoner_depth: int = 2,
        h_cycles: int = 3,
        l_cycles: int = 2,
        decoder_refine_steps: int = 4,
        pred_depth: int = 2,
        symbolic_train_cap: float = 0.5,
        symbolic_infer_cap: float = 1.0,
        # Legacy compat: these are silently ignored if present
        reasoner_max_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = IJepaEncoder2DRoPE(num_colors=num_colors + 1, dim=dim, depth=depth, heads=heads)

        # EMA target encoder for proper I-JEPA (no gradients)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # I-JEPA predictor (lightweight)
        self.ijepa_predictor = IJepaPredictor(dim=dim, depth=pred_depth, heads=heads)

        self.rule = CrossAttentionRuleExtractor(dim=dim, heads=heads)
        self.router = NoveltyRouter(dim=dim)
        self.reasoner = TrmHrmReasoner(
            dim=dim,
            depth=reasoner_depth,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
            heads=heads,
        )
        self.decoder = IterativeRefinementDecoder(dim=dim, num_colors=num_colors, refine_steps=decoder_refine_steps)
        self.num_colors = num_colors
        self.symbolic_train_cap = symbolic_train_cap
        self.symbolic_infer_cap = symbolic_infer_cap

    @torch.no_grad()
    def update_target_encoder(self, ema_decay: float = 0.996) -> None:
        """EMA update of target encoder from online encoder."""
        for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(ema_decay).add_((1.0 - ema_decay) * p_online.data)

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
        return self.reasoner.adapt_structure(
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
        """grid: [H, W] -> latent: [1, H*W, D]"""
        return self.encoder(grid.unsqueeze(0))

    def compute_ijepa_loss(self, grid: torch.Tensor, mask_ratio: float = 0.4) -> torch.Tensor:
        """Proper I-JEPA: predict target encoder latents from context encoder + predictor."""
        grid_batch = grid.unsqueeze(0)  # [1, H, W]
        b, h, w = grid_batch.shape
        t = h * w

        with torch.no_grad():
            target_latent = self.target_encoder(grid_batch)
            target_latent = F.layer_norm(target_latent, (target_latent.shape[-1],))

        context_latent = self.encoder(grid_batch)

        num_target = max(1, int(t * mask_ratio))
        target_indices = torch.randperm(t, device=grid.device)[:num_target]

        pred = self.ijepa_predictor(context_latent, target_indices, h, w)
        target = target_latent[:, target_indices]

        return F.smooth_l1_loss(pred, target)

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

        # I-JEPA loss: computed on test input only (efficient)
        if self.training:
            ijepa_loss = self.compute_ijepa_loss(test_input)
        else:
            ijepa_loss = torch.zeros((), device=test_input.device)

        rule_out = self.rule(
            test_latent=test_latent,
            train_input_latents=train_in_latents,
            train_output_latents=train_out_latents,
        )
        router_out = self.router(rule_out.test_conditioned_latent)

        h, w = test_input.shape
        reasoner_out = self.reasoner(
            rule_out.test_conditioned_latent,
            route_control=router_out.route_control,
            h=h,
            w=w,
        )
        decode_out = self.decoder(reasoner_out.latent, route_control=router_out.route_control)

        neural_logits = decode_out.logits.reshape(1, h, w, -1).permute(0, 3, 1, 2)

        sym = fit_symbolic_rule(train_inputs=train_inputs, train_outputs=train_outputs, test_input=test_input)
        sym_one_hot = torch.nn.functional.one_hot(sym.grid.clamp(0, self.num_colors - 1), num_classes=self.num_colors).float()
        sym_logits = sym_one_hot.permute(2, 0, 1).unsqueeze(0) * 6.0
        sym_w = float(max(0.0, min(1.0, sym.confidence)))
        cap = self.symbolic_train_cap if self.training else self.symbolic_infer_cap
        sym_w = float(min(sym_w, max(0.0, min(1.0, cap))))
        test_logits = (1.0 - sym_w) * neural_logits + sym_w * sym_logits

        return UnifiedForwardOutput(
            test_logits=test_logits,
            ijepa_loss=ijepa_loss,
            router=router_out,
            reasoner=reasoner_out,
            decoder=decode_out,
            test_shape=test_input.shape,
            symbolic_confidence=torch.tensor(sym_w, device=test_input.device),
            symbolic_rule=sym.rule_name,
        )
