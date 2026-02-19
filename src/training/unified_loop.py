from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque
import json
from pathlib import Path
import random

import torch

from src.data.arc_simulator import ArcEpisode
from src.models.unified_model import UnifiedArcModel
from src.training.losses import compute_unified_loss


@dataclass
class TrainConfig:
    steps: int = 2000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cpu"
    replay_ratio: float = 0.3
    replay_capacity: int = 1024
    log_every: int = 50
    sft_warmup_steps: int = 200
    min_sft_weight: float = 0.25
    sft_refresh_interval: int = 0
    sft_refresh_span: int = 0
    sft_refresh_min: float = 0.6
    routing_explore_steps: int = 300
    routing_explore_floor: float = 0.15
    routing_entropy_bonus: float = 0.02
    metrics_jsonl_path: str = ""
    rolling_window: int = 50
    show_progress_bar: bool = True
    adapt_structure: bool = True
    adapt_interval: int = 1
    grow_threshold: float = 0.55
    prune_threshold: float = 0.15
    structure_ema_momentum: float = 0.9
    hard_expert_cap: int = 6
    target_expert_cap: int = 4
    growth_force_until_step: int = 0
    growth_force_interval: int = 0
    growth_force_exact_threshold: float = 0.15
    prune_cooldown_steps: int = 0
    expert_budget_coeff: float = 0.25
    compute_penalty_coeff: float = 0.01
    reward_pixel_coeff: float = 0.2
    reward_exact_coeff: float = 1.0
    # New: EMA for I-JEPA target encoder
    ema_decay: float = 0.996
    # New: Q-halt loss weight
    w_qhalt: float = 0.5


class UnifiedTrainer:
    def __init__(self, model: UnifiedArcModel, cfg: TrainConfig) -> None:
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.failure_replay: deque[ArcEpisode] = deque(maxlen=cfg.replay_capacity)
        self.rng = random.Random(0)
        self.recent: deque[Dict[str, float]] = deque(maxlen=max(10, cfg.rolling_window))
        self.cumulative_solved = 0
        self.cumulative_total = 0
        if cfg.metrics_jsonl_path:
            Path(cfg.metrics_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cfg.metrics_jsonl_path, "w", encoding="utf-8") as f:
                f.write("")

    def _pick_episode(self, fresh_ep: ArcEpisode) -> ArcEpisode:
        if self.failure_replay and self.rng.random() < self.cfg.replay_ratio:
            return self.rng.choice(list(self.failure_replay))
        return fresh_ep

    def train_step(self, episode: ArcEpisode, step: int) -> Dict[str, float]:
        self.model.train()
        self.optim.zero_grad(set_to_none=True)

        out = self.model(
            train_inputs=episode.train_inputs,
            train_outputs=episode.train_outputs,
            test_input=episode.test_input,
        )
        loss = compute_unified_loss(
            out,
            episode.test_output,
            step=step,
            total_steps=self.cfg.steps,
            sft_warmup_steps=self.cfg.sft_warmup_steps,
            min_sft_weight=self.cfg.min_sft_weight,
            sft_refresh_interval=self.cfg.sft_refresh_interval,
            sft_refresh_span=self.cfg.sft_refresh_span,
            sft_refresh_min=self.cfg.sft_refresh_min,
            routing_explore_steps=self.cfg.routing_explore_steps,
            routing_explore_floor=self.cfg.routing_explore_floor,
            routing_entropy_bonus=self.cfg.routing_entropy_bonus,
            expert_budget_coeff=self.cfg.expert_budget_coeff,
            compute_penalty_coeff=self.cfg.compute_penalty_coeff,
            reward_pixel_coeff=self.cfg.reward_pixel_coeff,
            reward_exact_coeff=self.cfg.reward_exact_coeff,
            w_qhalt=self.cfg.w_qhalt,
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optim.step()

        # EMA update of I-JEPA target encoder
        self.model.update_target_encoder(self.cfg.ema_decay)

        self.model.router.update_memory(out.reasoner.latent)

        grown = False
        pruned = False
        if self.cfg.adapt_structure and (step % max(1, self.cfg.adapt_interval) == 0):
            grown, pruned = self.model.adapt_structure(
                reward=float(loss.reward.cpu()),
                novelty=float(out.router.novelty.mean().detach().cpu()),
                uncertainty=float(out.router.uncertainty.mean().detach().cpu()),
                hard_cap=self.cfg.hard_expert_cap,
                target_cap=self.cfg.target_expert_cap,
                grow_threshold=self.cfg.grow_threshold,
                prune_threshold=self.cfg.prune_threshold,
                ema_momentum=self.cfg.structure_ema_momentum,
                prune_cooldown_steps=self.cfg.prune_cooldown_steps,
            )
            if (
                not grown
                and self.cfg.growth_force_until_step > 0
                and step <= self.cfg.growth_force_until_step
                and self.cfg.growth_force_interval > 0
                and step % self.cfg.growth_force_interval == 0
                and float(loss.exact_match.cpu()) <= self.cfg.growth_force_exact_threshold
                and self.model.reasoner.active_experts < min(self.cfg.hard_expert_cap, self.model.reasoner.max_experts)
            ):
                self.model.reasoner.active_experts += 1
                grown = True

        if loss.exact_match.item() < 1.0:
            self.failure_replay.append(episode)

        if loss.eff_sft_weight.item() >= 0.67:
            path_chosen = "SFT"
        elif loss.eff_sft_weight.item() <= 0.33:
            path_chosen = "RL"
        else:
            path_chosen = "HYBRID"

        return {
            "total": float(loss.total.detach().cpu()),
            "sft": float(loss.l_sft.detach().cpu()),
            "qhalt": float(loss.l_qhalt.detach().cpu()),
            "ijepa": float(loss.l_ijepa.detach().cpu()),
            "reward": float(loss.reward.cpu()),
            "exact": float(loss.exact_match.cpu()),
            "alpha_sft": float(out.router.alpha_sft.mean().detach().cpu()),
            "alpha_rl": float(out.router.alpha_rl.mean().detach().cpu()),
            "eff_sft_weight": float(loss.eff_sft_weight.cpu()),
            "eff_rl_weight": float(loss.eff_rl_weight.cpu()),
            "novelty": float(out.router.novelty.mean().detach().cpu()),
            "uncertainty": float(out.router.uncertainty.mean().detach().cpu()),
            "explore_strength": float(loss.explore_strength.cpu()),
            "steps_reasoner": float(out.reasoner.steps_used),
            "steps_refine": float(out.decoder.refinement_steps),
            "active_experts": float(out.reasoner.active_experts),
            "grew": float(grown),
            "pruned": float(pruned),
            "path_chosen_sft": float(path_chosen == "SFT"),
            "path_chosen_rl": float(path_chosen == "RL"),
            "path_chosen_hybrid": float(path_chosen == "HYBRID"),
            "symbolic_confidence": float(out.symbolic_confidence.detach().cpu()),
        }

    def _summarize_recent(self) -> Dict[str, float]:
        if not self.recent:
            return {}
        keys = [
            "total",
            "exact",
            "alpha_sft",
            "alpha_rl",
            "eff_sft_weight",
            "eff_rl_weight",
            "novelty",
            "uncertainty",
            "explore_strength",
            "active_experts",
            "symbolic_confidence",
        ]
        out = {}
        n = len(self.recent)
        for k in keys:
            out[k] = sum(m[k] for m in self.recent) / n
        out["sft_dominant_frac"] = sum(1.0 for m in self.recent if m["eff_sft_weight"] >= m["eff_rl_weight"]) / n
        out["grow_events"] = sum(m.get("grew", 0.0) for m in self.recent)
        out["prune_events"] = sum(m.get("pruned", 0.0) for m in self.recent)
        out["path_sft_frac"] = sum(m.get("path_chosen_sft", 0.0) for m in self.recent) / n
        out["path_rl_frac"] = sum(m.get("path_chosen_rl", 0.0) for m in self.recent) / n
        out["path_hybrid_frac"] = sum(m.get("path_chosen_hybrid", 0.0) for m in self.recent) / n
        return out

    def _write_metric(self, metric: Dict[str, float]) -> None:
        if not self.cfg.metrics_jsonl_path:
            return
        with open(self.cfg.metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric) + "\n")

    def train_stream(self, stream) -> List[Dict[str, float]]:
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        logs: List[Dict[str, float]] = []
        pbar = None
        if tqdm is not None and self.cfg.show_progress_bar:
            pbar = tqdm(total=self.cfg.steps, dynamic_ncols=True, desc="training")

        for step, fresh_episode in enumerate(stream, start=1):
            episode = self._pick_episode(fresh_episode)
            m = self.train_step(episode, step=step)
            m["step"] = step
            self.cumulative_total += 1
            self.cumulative_solved += int(m["exact"] >= 1.0)
            m["cumulative_solved_pct"] = 100.0 * self.cumulative_solved / max(1, self.cumulative_total)
            m["rolling_solved_pct"] = 100.0 * m["exact"]
            logs.append(m)
            self.recent.append(m)
            self._write_metric(m)
            if pbar is not None:
                pbar.update(1)
                if step % max(1, self.cfg.log_every // 2) == 0:
                    avg = self._summarize_recent()
                    pbar.set_postfix(
                        {
                            "loss": f"{avg.get('total', m['total']):.3f}",
                            "exact": f"{avg.get('exact', m['exact']):.3f}",
                            "r_sft": f"{avg.get('alpha_sft', m['alpha_sft']):.2f}",
                            "r_rl": f"{avg.get('alpha_rl', m['alpha_rl']):.2f}",
                            "e_sft": f"{avg.get('eff_sft_weight', m['eff_sft_weight']):.2f}",
                            "e_rl": f"{avg.get('eff_rl_weight', m['eff_rl_weight']):.2f}",
                            "explore": f"{avg.get('explore_strength', m['explore_strength']):.2f}",
                            "solved%": f"{m['cumulative_solved_pct']:.1f}",
                            "xperts": f"{avg.get('active_experts', m['active_experts']):.1f}",
                            "path": f"S{avg.get('path_sft_frac', 0.0):.2f}/R{avg.get('path_rl_frac', 0.0):.2f}/H{avg.get('path_hybrid_frac', 0.0):.2f}",
                            "sym": f"{avg.get('symbolic_confidence', m['symbolic_confidence']):.2f}",
                        }
                    )
            if step % self.cfg.log_every == 0:
                avg = self._summarize_recent()
                print(
                    f"step={step} total={avg.get('total', m['total']):.4f} exact={avg.get('exact', m['exact']):.3f} "
                    f"router_sft={avg.get('alpha_sft', m['alpha_sft']):.3f} router_rl={avg.get('alpha_rl', m['alpha_rl']):.3f} "
                    f"eff_sft={avg.get('eff_sft_weight', m['eff_sft_weight']):.3f} eff_rl={avg.get('eff_rl_weight', m['eff_rl_weight']):.3f} "
                    f"explore={avg.get('explore_strength', m['explore_strength']):.3f} "
                    f"solved%={m['cumulative_solved_pct']:.2f} "
                    f"xperts={avg.get('active_experts', m['active_experts']):.2f} "
                    f"grow={avg.get('grow_events', 0.0):.0f} prune={avg.get('prune_events', 0.0):.0f} "
                    f"path[S/R/H]={avg.get('path_sft_frac', 0.0):.2f}/{avg.get('path_rl_frac', 0.0):.2f}/{avg.get('path_hybrid_frac', 0.0):.2f} "
                    f"sym={avg.get('symbolic_confidence', m['symbolic_confidence']):.2f} "
                    f"sft_dominant={avg.get('sft_dominant_frac', 0.0):.2f}"
                )
            if step >= self.cfg.steps:
                break
        if pbar is not None:
            pbar.close()
        return logs


@torch.no_grad()
def run_inference(model: UnifiedArcModel, episode: ArcEpisode) -> torch.Tensor:
    model.eval()
    out = model(
        train_inputs=episode.train_inputs,
        train_outputs=episode.train_outputs,
        test_input=episode.test_input,
    )
    return out.test_logits.argmax(dim=1).squeeze(0)
