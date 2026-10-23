"""Publish the fine-tuned GGUF to the Hub and the app to a Docker Space.

    uv run --with huggingface_hub hf auth login          # once, with a write token
    uv run --with huggingface_hub python deploy/publish.py [--private]

Model repo: <user>/tripmate-1.5b-gguf. Space: <user>/tripmate (Docker, free CPU).
The Space builds the Dockerfile; Space variables are passed to the build as build args,
so GGUF_URL tells it where to download the model from.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "finetune" / "tripmate-1.5b-q4_k_m.gguf"
SPACE_FILES = [
    "Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock",
    "src", "app", "deploy/start.sh",
    "data/destinations.yaml", "data/kb", "data/seeds",
]  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--private", action="store_true")
    p.add_argument("--model-repo", default="tripmate-1.5b-gguf")
    p.add_argument("--space", default="tripmate")
    args = p.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    model_id, space_id = f"{user}/{args.model_repo}", f"{user}/{args.space}"

    api.create_repo(model_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=ROOT / "deploy" / "MODEL_CARD.md", path_in_repo="README.md", repo_id=model_id
    )
    api.upload_file(path_or_fileobj=GGUF, path_in_repo=GGUF.name, repo_id=model_id)
    gguf_url = f"https://huggingface.co/{model_id}/resolve/main/{GGUF.name}"
    print("model:", f"https://huggingface.co/{model_id}")

    api.create_repo(
        space_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True
    )
    api.add_space_variable(space_id, "GGUF_URL", gguf_url)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for rel in SPACE_FILES:
            src, dst = ROOT / rel, stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(src, dst)
        shutil.copy2(ROOT / "deploy" / "SPACE_README.md", stage / "README.md")
        api.upload_folder(folder_path=stage, repo_id=space_id, repo_type="space")
    print("space:", f"https://huggingface.co/spaces/{space_id}  (first build takes ~10 min)")


if __name__ == "__main__":
    main()
