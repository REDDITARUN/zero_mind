from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from collections import deque
import json
import math
from pathlib import Path
import random

import torch

from src.data.arc_simulator import ArcEpisode
from src.models.unified_model import UnifiedArcModel
from src.training.losses import compute_unified_loss


@dataclass
class TrainConfig:
    steps: int = 2000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cpu"
    replay_ratio: float = 0.3
    replay_capacity: int = 1024
    log_every: int = 50
    metrics_jsonl_path: str = ""
    rolling_window: int = 50
    show_progress_bar: bool = True

    lr_warmup_steps: int = 500
    lr_min_ratio: float = 0.05

    grad_accum_steps: int = 4

    use_amp: bool = False
    compile_model: bool = False

    w_draft: float = 0.5
    w_aux: float = 0.3
    w_focus: float = 1.0
    w_consistency: float = 0.02
    label_smoothing: float = 0.05
    loss_clamp: float = 4.0

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

    # Legacy compat — accepted but ignored
    ema_decay: float = 0.996
    w_dino: float = 0.0
    w_shape: float = 0.0
    sft_warmup_steps: int = 0
    min_sft_weight: float = 0.0
    sft_refresh_interval: int = 0
    sft_refresh_span: int = 0
    sft_refresh_min: float = 0.0
    routing_explore_steps: int = 0
    routing_explore_floor: float = 0.0
    routing_entropy_bonus: float = 0.0
    reward_pixel_coeff: float = 0.0
    reward_exact_coeff: float = 1.0
    w_qhalt: float = 0.0


def _cosine_lr(step: int, warmup: int, total: int, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return max(min_ratio, step / max(1, warmup))
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


class UnifiedTrainer:
    def __init__(self, model: UnifiedArcModel, cfg: TrainConfig) -> None:
        self.model = model.to(cfg.device)
        self.cfg = cfg

        self._use_amp = cfg.use_amp and cfg.device.startswith("cuda")
        self._amp_dtype = torch.bfloat16

        if cfg.device.startswith("cuda"):
            try:
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            except (AttributeError, TypeError):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("CUDA optimizations: TF32 + cudnn.benchmark enabled")

        if cfg.compile_model and cfg.device.startswith("cuda"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("torch.compile enabled (reduce-overhead)")
            except Exception as e:
                print(f"torch.compile failed ({e}), using eager mode")

        self.optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.failure_replay: deque[ArcEpisode] = deque(maxlen=cfg.replay_capacity)
        self.rng = random.Random(0)
        self.recent: deque[Dict[str, float]] = deque(maxlen=max(10, cfg.rolling_window))
        self.cumulative_solved = 0
        self.cumulative_total = 0
        self._accum_count = 0
        if cfg.metrics_jsonl_path:
            Path(cfg.metrics_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cfg.metrics_jsonl_path, "w", encoding="utf-8") as f:
                f.write("")

    def _set_lr(self, step: int) -> float:
        mult = _cosine_lr(step, self.cfg.lr_warmup_steps, self.cfg.steps, self.cfg.lr_min_ratio)
        lr = self.cfg.lr * mult
        for pg in self.optim.param_groups:
            pg["lr"] = lr
        return lr

    def _pick_episode(self, fresh_ep: ArcEpisode) -> ArcEpisode:
        if self.failure_replay and self.rng.random() < self.cfg.replay_ratio:
            return self.rng.choice(list(self.failure_replay))
        return fresh_ep

    def train_step(self, episode: ArcEpisode, step: int) -> Dict[str, float]:
        self.model.train()
        current_lr = self._set_lr(step)

        device_type = "cuda" if self.cfg.device.startswith("cuda") else "cpu"
        with torch.amp.autocast(device_type, dtype=self._amp_dtype, enabled=self._use_amp):
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
                lr_warmup_steps=self.cfg.lr_warmup_steps,
                w_draft=self.cfg.w_draft,
                w_aux=self.cfg.w_aux,
                w_focus=self.cfg.w_focus,
                w_consistency=self.cfg.w_consistency,
                label_smoothing=self.cfg.label_smoothing,
                loss_clamp=self.cfg.loss_clamp,
                expert_budget_coeff=self.cfg.expert_budget_coeff,
                compute_penalty_coeff=self.cfg.compute_penalty_coeff,
            )

        scaled_loss = loss.total / self.cfg.grad_accum_steps
        scaled_loss.backward()
        self._accum_count += 1

        if self._accum_count >= self.cfg.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
            self._accum_count = 0

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
                and self.model.content_reasoner.active_experts < min(
                    self.cfg.hard_expert_cap, self.model.content_reasoner.max_experts
                )
            ):
                self.model.content_reasoner.active_experts += 1
                grown = True

        if loss.exact_match.item() < 1.0:
            self.failure_replay.append(episode)

        draft_acc = float(
            (out.draft_logits.argmax(dim=1).squeeze(0) == episode.test_output).float().mean().cpu()
        )

        return {
            "total": float(loss.total.detach().cpu()),
            "correct": float(loss.l_correct.detach().cpu()),
            "draft": float(loss.l_draft.detach().cpu()),
            "aux": float(loss.l_aux.detach().cpu()),
            "focus": float(loss.l_focus.detach().cpu()),
            "consistency": float(loss.l_consistency.detach().cpu()),
            "reward": float(loss.reward.cpu()),
            "exact": float(loss.exact_match.cpu()),
            "pixel_acc": float((out.test_logits.argmax(dim=1).squeeze(0) == episode.test_output).float().mean().cpu()),
            "draft_acc": draft_acc,
            "lr": current_lr,
            "alpha_sft": float(out.router.alpha_sft.mean().detach().cpu()),
            "alpha_rl": float(out.router.alpha_rl.mean().detach().cpu()),
            "novelty": float(out.router.novelty.mean().detach().cpu()),
            "uncertainty": float(out.router.uncertainty.mean().detach().cpu()),
            "steps_refine": float(out.decoder.refinement_steps),
            "active_experts": float(out.reasoner.active_experts),
            "grew": float(grown),
            "pruned": float(pruned),
            "symbolic_confidence": float(out.symbolic_confidence.detach().cpu()),
        }

    def _summarize_recent(self) -> Dict[str, float]:
        if not self.recent:
            return {}
        keys = [
            "total", "correct", "draft", "aux", "exact", "pixel_acc",
            "draft_acc", "novelty", "uncertainty", "active_experts",
            "symbolic_confidence",
        ]
        out: Dict[str, float] = {}
        n = len(self.recent)
        for k in keys:
            out[k] = sum(m.get(k, 0.0) for m in self.recent) / n
        out["grow_events"] = sum(m.get("grew", 0.0) for m in self.recent)
        out["prune_events"] = sum(m.get("pruned", 0.0) for m in self.recent)
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

        self.optim.zero_grad(set_to_none=True)

        for step, fresh_episode in enumerate(stream, start=1):
            episode = self._pick_episode(fresh_episode)
            m = self.train_step(episode, step=step)
            m["step"] = step
            self.cumulative_total += 1
            self.cumulative_solved += int(m["exact"] >= 1.0)
            m["cumulative_solved_pct"] = 100.0 * self.cumulative_solved / max(1, self.cumulative_total)
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
                            "px": f"{avg.get('pixel_acc', m['pixel_acc']):.3f}",
                            "drft": f"{avg.get('draft_acc', m['draft_acc']):.3f}",
                            "fix": f"{avg.get('pixel_acc', m['pixel_acc']) - avg.get('draft_acc', m['draft_acc']):+.3f}",
                            "exact": f"{avg.get('exact', m['exact']):.3f}",
                            "solved%": f"{m['cumulative_solved_pct']:.1f}",
                            "xprt": f"{avg.get('active_experts', m['active_experts']):.0f}",
                            "lr": f"{m['lr']:.1e}",
                        }
                    )
            if step % self.cfg.log_every == 0:
                avg = self._summarize_recent()
                fix_delta = avg.get('pixel_acc', m['pixel_acc']) - avg.get('draft_acc', m['draft_acc'])
                print(
                    f"step={step} loss={avg.get('total', m['total']):.4f} "
                    f"correct={avg.get('correct', m['correct']):.3f} "
                    f"draft={avg.get('draft', m['draft']):.3f} "
                    f"focus={avg.get('focus', m.get('focus', 0)):.3f} "
                    f"px={avg.get('pixel_acc', m['pixel_acc']):.3f} "
                    f"drft={avg.get('draft_acc', m['draft_acc']):.3f} "
                    f"fix={fix_delta:+.3f} "
                    f"exact={avg.get('exact', m['exact']):.3f} "
                    f"solved%={m['cumulative_solved_pct']:.2f} "
                    f"xprt={avg.get('active_experts', m['active_experts']):.0f} "
                    f"lr={m['lr']:.2e}"
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
