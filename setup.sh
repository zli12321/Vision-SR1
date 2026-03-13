pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Step 2: Install vllm compatible version
pip install vllm==0.11.0

# Step 3: Verify torch didn't change
python -c "import torch; print(torch.__version__)"

# Step 4: Build flash-attn against this exact torch
MAX_JOBS=96 pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir

# Step 5: Install EasyR1
# cd /workspace/Self-Agent
pip install -e . --no-deps
pip install accelerate codetiming datasets liger-kernel mathruler numpy omegaconf pandas peft pillow pyarrow pylatexenc qwen-vl-utils ray[default] tensordict torchdata transformers wandb

pip install transformers==4.57.0

# Then run again. If it still fails, try:
# pip install transformers==4.54.0

pip install flask stopit pandas pyarrow matplotlib regex nltk scikit-learn plotly kaleido Pillow cairosvg

# Plotly/Kaleido requires Chrome + OS-level dependencies for fig.write_image()
plotly_get_chrome || echo "[setup] plotly_get_chrome not found, skipping Chrome install"
apt-get update -qq && apt-get install -y -qq \
    libnss3 libnspr4 \
    libatk1.0-0t64 libatk-bridge2.0-0t64 \
    libcups2t64 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libxkbcommon0 \
    libpango-1.0-0 libcairo2 \
    libasound2t64 \
    libatspi2.0-0t64 \
    2>/dev/null || echo "[setup] Some apt packages may have different names on your OS"

wandb login
