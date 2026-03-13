# # pip install -U huggingface_hub
# from huggingface_hub import snapshot_download

# # ---- fill these in ----
# repo_id   = "LMMs-Lab-Turtle/Vision-SR1-Cold-9K"   
# repo_type = "dataset"                 # "model" | "dataset" | "space"
# target    = "./LLaMA-Factory-Cold-Start/data"     # where to put the data
# revision  = None                   

# # If it's a private repo, set HF token:
# # import os; os.environ["HF_TOKEN"] = "hf_xxx"

# snapshot_download(
#     repo_id=repo_id,
#     repo_type=repo_type,
#     revision=revision,
#     local_dir=target,
#     local_dir_use_symlinks=False,  # set True to save space via symlinks
#     # Optional filters:
#     # allow_patterns=["subdir/**", "*.json"], 
#     # ignore_patterns=["*.pt", "*.bin"],
#     resume_download=True,          # continues partial downloads
#     max_workers=1                  # parallelism
# )
# print("Done!")

from huggingface_hub import snapshot_download
import time
import random
from requests.exceptions import HTTPError

def download_with_retry(repo_id, repo_type, target, max_retries=5):
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=target,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=1  # Reduce parallelism
            )
            print("Download completed successfully!")
            return
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
            else:
                # Add a shorter sleep for other errors too
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    wait_time = 5 + random.uniform(0, 2)  # 5-7 seconds
                    print(f"Error occurred: {e}")
                    print(f"Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    raise e
    
    raise Exception(f"Failed after {max_retries} retries")

# Usage
repo_id = "LMMs-Lab-Turtle/Vision-SR1-Cold-9K"
repo_type = "dataset"
target = "./LLaMA-Factory-Cold-Start/data"

download_with_retry(repo_id, repo_type, target)