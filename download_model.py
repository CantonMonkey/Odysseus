"""Download InternVL3-8B from ModelScope (faster on autodl CN network)."""
import os, subprocess, sys

MODEL_DIR = "/root/autodl-tmp/models/OpenGVLab/InternVL3-8B"

def main():
    if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print(f"Model already at {MODEL_DIR}, skipping download.")
        return

    os.makedirs("/root/autodl-tmp/models", exist_ok=True)

    # Try ModelScope first (faster on autodl CN servers)
    try:
        from modelscope import snapshot_download
        print("Downloading via ModelScope...")
        path = snapshot_download(
            "OpenGVLab/InternVL3-8B",
            cache_dir="/root/autodl-tmp/models",
        )
        print(f"Downloaded to: {path}")
        return
    except ImportError:
        pass
    except Exception as e:
        print(f"ModelScope failed: {e}, trying HuggingFace...")

    # Fallback: huggingface_hub
    from huggingface_hub import snapshot_download as hf_download
    print("Downloading via HuggingFace Hub...")
    path = hf_download(
        "OpenGVLab/InternVL3-8B",
        local_dir=MODEL_DIR,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
    )
    print(f"Downloaded to: {path}")

if __name__ == "__main__":
    main()
