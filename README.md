# ZERO MIND — RL Agent for ARC-AGI

A reinforcement learning agent that learns to solve [ARC-AGI](https://arcprize.org/) tasks via **autoregressive grid generation**. The agent predicts the output grid dimensions (height, width) then fills each cell sequentially, trained end-to-end with PPO.

## Architecture

- **Environment** (`arc_env/gen_env.py`): Gymnasium env where the agent generates grids in 3 phases — predict H → predict W → fill cells left-to-right, top-to-bottom.
- **Policy** (`arc_policy.py`): Encoder-decoder Transformer. The encoder processes tokenized demo pairs + test input; the decoder autoregressively generates the output grid.
- **Training** (`train.py`): Batched PPO with GAE, curriculum sampling, gradient clipping, and per-task checkpointing.

## Quick Setup

```bash
git clone https://github.com/REDDITARUN/zero_mind.git
cd zero_mind

# Get ARC data
git clone https://github.com/fchollet/ARC-AGI.git
git clone https://github.com/arcprize/ARC-AGI-2.git

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## Usage

### Train on a single task (sanity check)

```bash
python train.py --single-task 52 --max-episodes 2000 --d-model 64 --n-heads 4 --n-enc-layers 2 --n-dec-layers 2
```

### Train on all 400 tasks

```bash
python train.py --batch-size 16 --max-episodes 100000
```

### Interactive play

```bash
python play_arc.py --show-target
```

### Visualize a trained agent

```bash
python visualize_run.py
```

## Project Structure

```
├── arc_env/
│   ├── gen_env.py          # Autoregressive grid generation environment
│   ├── env.py              # Original canvas-based environment (legacy)
│   ├── augment.py          # Task loading + geometric augmentation
│   ├── curriculum.py       # Adaptive task sampler
│   ├── rewards.py          # Reward calculator (legacy env)
│   ├── renderer.py         # ANSI grid rendering
│   └── policy.py           # CNN+Transformer policy (legacy env)
├── arc_policy.py           # Encoder-decoder Transformer policy
├── train.py                # PPO training loop
├── train_single_task.py    # Single-task training (legacy env)
├── play_arc.py             # Interactive play
├── visualize_run.py        # GIF generation from checkpoints
└── requirements.txt
```

## How It Works

1. **Tokenization**: Demo input/output pairs and the test input are converted into a flat token sequence with special separator tokens and 2D positional embeddings.
2. **Encoding**: A Transformer encoder processes the full context bidirectionally.
3. **Decoding**: The decoder predicts output height (1-30), then width (1-30), then each cell color (0-9) autoregressively with causal masking.
4. **Reward**: +1 for correct H/W, per-cell shaped rewards during generation, +5 terminal bonus for perfect solve.
5. **PPO**: Standard clipped PPO with GAE, entropy bonus, and gradient clipping trains the policy end-to-end.
