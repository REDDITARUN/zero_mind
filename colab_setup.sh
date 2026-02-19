#!/bin/bash
set -e

echo "=== ZERO_MIND Colab Setup ==="

# Detect if we're already inside the repo
if [ -f "colab_setup.sh" ] && [ -f "pyproject.toml" ]; then
    echo "Already inside zero_mind repo."
else
    echo "ERROR: Run this script from inside the zero_mind repo directory."
    echo "  git clone https://github.com/REDDITARUN/zero_mind.git"
    echo "  cd zero_mind"
    echo "  bash colab_setup.sh"
    exit 1
fi

# 2. Install deps
pip install -q -r requirements.txt

# 3. Clone ARC-AGI data
mkdir -p References
if [ ! -d "References/ARC-AGI" ]; then
    echo "Cloning ARC-AGI dataset..."
    git clone --depth 1 https://github.com/fchollet/ARC-AGI.git References/ARC-AGI
fi

# 4. Create data/checkpoints/logs dirs
mkdir -p data checkpoints logs reports

# 5. Build augmented episodes
if [ ! -f "data/arc_400_tasks_12k_episodes.jsonl" ]; then
    echo "Building augmented episodes (400 tasks x 30 episodes)..."
    python scripts/build_augmented_arc_episodes.py \
        --arc_data_dir References/ARC-AGI/data \
        --split training \
        --num_tasks 400 \
        --episodes_per_task 30 \
        --out_file data/arc_400_tasks_12k_episodes.jsonl
    echo "Episodes built: $(wc -l < data/arc_400_tasks_12k_episodes.jsonl) lines"
else
    echo "Episodes file already exists, skipping build."
fi

# 6. Verify GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = getattr(torch.cuda.get_device_properties(0), 'total_memory', None) or getattr(torch.cuda.get_device_properties(0), 'total_mem', 0)
    print(f'Memory: {mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected! Training will be slow.')
"

# 7. Quick smoke test
echo "Running smoke test..."
python -c "
import torch
from src.models.unified_model import UnifiedArcModel
from src.training.losses import compute_unified_loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
m = UnifiedArcModel(num_colors=10, dim=64, depth=2, heads=4, reasoner_depth=1, h_cycles=2, l_cycles=2, shape_reasoner_depth=1, shape_h_cycles=1, shape_l_cycles=1, decoder_refine_steps=2)
m = m.to(device)
ti = [torch.randint(0,10,(5,5),device=device)]
to = [torch.randint(0,10,(5,5),device=device)]
tx = torch.randint(0,10,(5,5),device=device)
m.train()
out = m(ti,to,tx)
tgt = torch.randint(0,10,(5,5),device=device)
loss = compute_unified_loss(out, tgt, step=1, total_steps=10)
loss.total.backward()
print(f'Smoke test PASSED on {device} | loss={loss.total.item():.3f}')
"

echo ""
echo "=== Setup complete! ==="
echo "To train:"
echo "  python scripts/run_train_with_eval.py --train_config configs/colab_cuda.yaml --eval_config configs/colab_eval.yaml --eval_every 4000 --eval_steps 200 --eval_pass_k 3"
echo ""
echo "Or quick (no eval checkpoints):"
echo "  python scripts/run_train.py --config configs/colab_cuda.yaml"
