pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install "vllm==0.8.4"

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install git+https://github.com/huggingface/transformers.git@1931a351408dbd1d0e2c4d6d7ee0eb5e8807d7bf

pip install -e .

pip uninstall torch flash-attn

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip uninstall torch vllm

pip install "vllm==0.8.4"

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install git+https://github.com/huggingface/transformers.git@1931a351408dbd1d0e2c4d6d7ee0eb5e8807d7bf

pip install transformers==4.49.0

if ! wandb status 2>/dev/null | grep -q "Logged in"; then
  echo "⚠️  You are not logged in to W&B."
  read -rs -p "Enter your W&B API key: " WANDB_KEY   # -s hides input
  echo                                                # newline after prompt
  wandb login "$WANDB_KEY"
fi
