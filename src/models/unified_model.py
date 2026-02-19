from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from src.models.ijepa_rope import IJepaEncoder2DRoPE, DINOHead, dino_loss
from src.models.cross_attention_rule import CrossAttentionRuleExtractor
from src.models.router import NoveltyRouter, RouterOutput
from src.models.reasoner_trm_hrm import TrmHrmReasoner, ReasonerOutput
from src.models.decoder_refine import IterativeRefinementDecoder, DecodeOutput
from src.models.symbolic_primitives import fit_symbolic_rule


@dataclass
class UnifiedForwardOutput:
    test_logits: torch.Tensor
    aux_logits: torch.Tensor
    dino_loss: torch.Tensor
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
        use_mamba: bool = False,
        dino_hidden: int = 512,
        dino_out: int = 256,
        symbolic_train_cap: float = 0.5,
        symbolic_infer_cap: float = 1.0,
        # Legacy compat: silently ignored
        pred_depth: int | None = None,
        reasoner_max_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = IJepaEncoder2DRoPE(
            num_colors=num_colors + 1, dim=dim, depth=depth, heads=heads, use_mamba=use_mamba,
        )

        # DINO: EMA target encoder + projection heads
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.student_head = DINOHead(dim=dim, hidden_dim=dino_hidden, out_dim=dino_out)
        self.teacher_head = copy.deepcopy(self.student_head)
        for p in self.teacher_head.parameters():
            p.requires_grad = False
        self.register_buffer("dino_center", torch.zeros(1, dino_out))

        self.rule = CrossAttentionRuleExtractor(dim=dim, heads=heads)

        # Router kept for novelty/uncertainty metrics + structure adaptation
        self.router = NoveltyRouter(dim=dim)

        self.reasoner = TrmHrmReasoner(
            dim=dim,
            depth=reasoner_depth,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
            heads=heads,
        )
        self.decoder = IterativeRefinementDecoder(dim=dim, num_colors=num_colors, refine_steps=decoder_refine_steps)

        # Auxiliary direct readout (shortcut gradient path for encoder)
        self.aux_readout = nn.Linear(dim, num_colors)

        self.num_colors = num_colors
        self.symbolic_train_cap = symbolic_train_cap
        self.symbolic_infer_cap = symbolic_infer_cap

    @torch.no_grad()
    def update_target_ema(self, ema_decay: float = 0.996) -> None:
        """EMA update of target encoder + teacher head."""
        for p_s, p_t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(ema_decay).add_((1.0 - ema_decay) * p_s.data)
        for p_s, p_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            p_t.data.mul_(ema_decay).add_((1.0 - ema_decay) * p_s.data)

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
        return self.encoder(grid.unsqueeze(0))

    def compute_dino_loss(self, grid: torch.Tensor, center_momentum: float = 0.9) -> torch.Tensor:
        """DINO self-distillation loss on the test input grid."""
        grid_batch = grid.unsqueeze(0)

        with torch.no_grad():
            teacher_latent = self.target_encoder(grid_batch)
            teacher_pooled = teacher_latent.mean(dim=1)
            teacher_proj = self.teacher_head(teacher_pooled)

        student_latent = self.encoder(grid_batch)
        student_pooled = student_latent.mean(dim=1)
        student_proj = self.student_head(student_pooled)

        loss = dino_loss(student_proj, teacher_proj, self.dino_center)

        with torch.no_grad():
            self.dino_center = (
                self.dino_center * center_momentum
                + teacher_proj.mean(dim=0, keepdim=True) * (1.0 - center_momentum)
            )

        return loss

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

        # DINO loss (training only)
        if self.training:
            d_loss = self.compute_dino_loss(test_input)
        else:
            d_loss = torch.zeros((), device=test_input.device)

        rule_out = self.rule(
            test_latent=test_latent,
            train_input_latents=train_in_latents,
            train_output_latents=train_out_latents,
        )

        # Router: runs for metrics/structure adaptation, doesn't control compute
        router_out = self.router(rule_out.test_conditioned_latent)

        h, w = test_input.shape
        reasoner_out = self.reasoner(rule_out.test_conditioned_latent, h=h, w=w)
        decode_out = self.decoder(reasoner_out.latent)

        neural_logits = decode_out.logits.reshape(1, h, w, -1).permute(0, 3, 1, 2)

        # Auxiliary direct readout (shortcut gradient path)
        aux_logits_raw = self.aux_readout(rule_out.test_conditioned_latent)
        aux_logits = aux_logits_raw.reshape(1, h, w, -1).permute(0, 3, 1, 2)

        sym = fit_symbolic_rule(train_inputs=train_inputs, train_outputs=train_outputs, test_input=test_input)
        sym_one_hot = torch.nn.functional.one_hot(sym.grid.clamp(0, self.num_colors - 1), num_classes=self.num_colors).float()
        sym_logits = sym_one_hot.permute(2, 0, 1).unsqueeze(0) * 6.0
        sym_w = float(max(0.0, min(1.0, sym.confidence)))
        cap = self.symbolic_train_cap if self.training else self.symbolic_infer_cap
        sym_w = float(min(sym_w, max(0.0, min(1.0, cap))))
        test_logits = (1.0 - sym_w) * neural_logits + sym_w * sym_logits

        return UnifiedForwardOutput(
            test_logits=test_logits,
            aux_logits=aux_logits,
            dino_loss=d_loss,
            router=router_out,
            reasoner=reasoner_out,
            decoder=decode_out,
            test_shape=test_input.shape,
            symbolic_confidence=torch.tensor(sym_w, device=test_input.device),
            symbolic_rule=sym.rule_name,
        )
