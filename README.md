# ZERO_MIND — Brain-Inspired ARC-AGI Solver

A unified architecture for ARC-AGI tasks that goes beyond reward-only learning, combining self-supervised representation learning, hierarchical latent-space reasoning, and diffusion-style self-correction in a single end-to-end system.

## Architecture

- **I-JEPA Encoder** — Self-supervised latent representations via EMA target encoder + masked prediction with 2D RoPE
- **Cross-Attention Rule Extractor** — Infers transformation rules from few input/output example pairs
- **TRM/HRM Reasoner** — Hierarchical H/L-cycle iterative reasoning in latent space with adaptive halting (Q-head) and dynamic expert capacity
- **Diffusion Decoder** — Iterative refinement for self-correcting grid generation
- **Novelty Router** — Automatically switches between SFT (novel tasks) and RL (familiar tasks) objectives
- **Neurosymbolic Primitives** — Blends symbolic rule predictions with neural output
- **Adaptive Network** — Grows/prunes expert capacity based on task complexity; smaller network = more reward

## Quick Start (Local)

```bash
pip install -r requirements.txt
python scripts/run_train.py --config configs/base.yaml
```

## Colab / GPU Training

One-liner to set up and run on Colab (paste in terminal):

```bash
!git clone https://github.com/REDDITARUN/zero_mind.git && cd zero_mind && bash colab_setup.sh
```

Or step by step:

```bash
# 1. Clone and setup
git clone https://github.com/REDDITARUN/zero_mind.git
cd zero_mind
bash colab_setup.sh

# 2. Train (24k steps, ~2-3 hrs on T4)
python scripts/run_train_with_eval.py \
  --train_config configs/colab_cuda.yaml \
  --eval_config configs/colab_eval.yaml \
  --eval_every 4000 \
  --eval_steps 200 \
  --eval_pass_k 3

# 3. Evaluate checkpoint
python scripts/run_eval.py \
  --config configs/colab_eval.yaml \
  --checkpoint checkpoints/unified_arc_colab.pt \
  --max_eval_steps 400 \
  --pass_k 3

# 4. Per-task breakdown
python scripts/eval_taskwise.py \
  --config configs/colab_eval.yaml \
  --checkpoint checkpoints/unified_arc_colab.pt \
  --max_eval_steps 400 \
  --out_file reports/taskwise_eval.jsonl
```

### Resume Training

```bash
python scripts/run_train_with_eval.py \
  --train_config configs/colab_cuda.yaml \
  --eval_config configs/colab_eval.yaml \
  --load_checkpoint checkpoints/unified_arc_colab.pt
```

## Training Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_train.py` | Basic training loop |
| `scripts/run_train_with_eval.py` | Training with periodic ARC eval + checkpoints |
| `scripts/run_eval.py` | Evaluate checkpoint (episode or task-level with pass@K) |
| `scripts/eval_taskwise.py` | Per-task eval report with router/symbolic diagnostics |
| `scripts/run_infer.py` | Run inference on ARC tasks |
| `scripts/visualize_inference.py` | Save human-readable inference previews |
| `scripts/build_augmented_arc_episodes.py` | Build augmented episode dataset from ARC JSON |
| `scripts/summarize_metrics.py` | Summarize training metrics JSONL |
| `scripts/arc_data_report.py` | Report on ARC data splits |

## Configs

| Config | Use Case |
|--------|----------|
| `configs/base.yaml` | Quick smoke test (CPU) |
| `configs/colab_cuda.yaml` | Full GPU training (Colab T4/A100) |
| `configs/colab_eval.yaml` | Evaluation on GPU |
| `configs/serious_arc_v2.yaml` | Long local training |
| `configs/arc_test_eval.yaml` | Evaluation on ARC evaluation split (CPU) |

## Monitoring Training

Progress bar shows: loss, exact match rate, cumulative solved %, active experts.

Per-step console logs include router weights (`router_sft`/`router_rl`), effective SFT/RL mix, and exploration state.

Full metrics trace: set `train.metrics_jsonl_path` in config, then:

```bash
python scripts/summarize_metrics.py --metrics_file logs/colab_cuda_metrics.jsonl
```

## Core Design

- Single forward pass activates all modules every step
- Router outputs soft `alpha_sft`, `alpha_rl` weights — no hard switching
- `L_total = eff_sft * L_sft + w_qhalt * L_qhalt + w_ijepa * L_ijepa + w_eff * L_eff + w_route * L_route + w_consistency * L_consistency`
- One backward pass updates encoder, cross-attention, router, reasoner, and decoder together
- I-JEPA target encoder updated via EMA (no extra gradient cost)
- Network grows/contracts experts based on performance — efficiency is rewarded
