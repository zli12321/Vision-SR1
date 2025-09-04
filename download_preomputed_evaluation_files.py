# pip install -U huggingface_hub
from huggingface_hub import snapshot_download

# ---- fill these in ----
repo_id   = "DingZhenDojoCat/Evaluation"   
repo_type = "dataset"                 # "model" | "dataset" | "space"
target    = "./Evaluation"     # where to put the data
revision  = None                   

# If it's a private repo, set HF token:
# import os; os.environ["HF_TOKEN"] = "hf_xxx"

snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    revision=revision,
    local_dir=target,
    local_dir_use_symlinks=False,  # set True to save space via symlinks
    # Optional filters:
    # allow_patterns=["subdir/**", "*.json"], 
    # ignore_patterns=["*.pt", "*.bin"],
    resume_download=True,          # continues partial downloads
    max_workers=8                  # parallelism
)
print("Done!")