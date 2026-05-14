from huggingface_hub import snapshot_download

# Base download folder - change this path if you want a different location
BASE_DIR = "models"

models = [
    # Grounding DINO
    ("IDEA-Research/grounding-dino-base",  "grounding-dino-base"),
    ("IDEA-Research/grounding-dino-tiny",  "grounding-dino-tiny"),
    # SAM 2
    # ("facebook/sam2.1-hiera-small",        "sam2-hiera-small"),
    # ("facebook/sam2.1-hiera-base-plus",    "sam2-hiera-base-plus"),
    # ("facebook/sam2.1-hiera-tiny",         "sam2-hiera-tiny"),
]



for repo_id, folder_name in models:
    local_dir = f"{BASE_DIR}/{folder_name}"
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id, local_dir=local_dir)
    print("  Done.")

print("All models downloaded.")


# These are the correct image-only SAM 2.1 checkpoints
# The ones you downloaded ealier may have been video variants

snapshot_download("facebook/sam2.1-hiera-small",
                  local_dir="models/sam2-hiera-small",
                  ignore_patterns=["*.pt"])   # skip the raw .pt file, use safetensors

snapshot_download("facebook/sam2.1-hiera-tiny",
                  local_dir="models/sam2-hiera-tiny",
                  ignore_patterns=["*.pt"])

snapshot_download("facebook/sam2.1-hiera-base-plus",
                  local_dir="models/sam2-hiera-base-plus",
                  ignore_patterns=["*.pt"])

