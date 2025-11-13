"""Upload trained LoRA adapters to the Hugging Face Hub."""

from __future__ import annotations

import argparse

from huggingface_hub import create_repo, login, upload_folder


def upload_adapter(repo_id: str, folder_path: str, commit_message: str, hf_token: str | None) -> None:
    if hf_token:
        login(token=hf_token)
    create_repo(repo_id, exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        commit_message=commit_message,
    )
    print(f"Uploaded {folder_path} to {repo_id}")


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a trained LoRA adapter folder to the HF Hub.")
    parser.add_argument("--repo-id", required=True, help="huggingface.co repo id, e.g. username/project-name")
    parser.add_argument("--folder-path", default="flux_lora", help="Folder that holds the adapter checkpoint.")
    parser.add_argument("--commit-message", default="Add FLUX LoRA adapter")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token (otherwise rely on cached login).")
    return parser.parse_args()


def main():
    args = parse_args()
    upload_adapter(args.repo_id, args.folder_path, args.commit_message, args.hf_token)


if __name__ == "__main__":
    main()
