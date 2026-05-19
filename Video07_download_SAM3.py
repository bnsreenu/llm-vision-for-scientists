# Get your token from: huggingface.co/settings/tokens

from huggingface_hub import snapshot_download

# Download SAM 3.1 (recommended, latest)
snapshot_download(
    repo_id="facebook/sam3.1",
    local_dir=r"C:\hf_models\sam3.1",
    token="YOUR_HF_TOKEN_HERE",
)

# Download SAM 3 base (optional, for comparison)
#snapshot_download(
#    repo_id="facebook/sam3",
#    local_dir=r"C:\hf_models\sam3",
#    token="YOUR_HF_TOKEN_HERE",
#)