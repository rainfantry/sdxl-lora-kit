"""
ComfyUI PNG metadata reader.
Usage: python read_metadata.py [folder_or_file_glob]

Examples:
    python read_metadata.py "C:\\Users\\<USER>\\Desktop\\ui\\output\\*.png"
    python read_metadata.py "C:\\Users\\<USER>\\Desktop\\ui\\output\\sks_woman_00042_.png"

Output: per-file table with seed, sampler, scheduler, cfg, steps, prompt, lora.
"""

from PIL import Image
import json
import sys
import glob
import os


def extract(png_path: str) -> dict:
    """Pull ComfyUI metadata from a PNG's text chunks."""
    out = {"file": os.path.basename(png_path)}
    try:
        img = Image.open(png_path)
        # ComfyUI writes two keys: 'workflow' (UI format) and 'prompt' (API format)
        workflow = img.info.get("workflow") or img.info.get("Workflow") or ""
        prompt_meta = img.info.get("prompt") or ""

        if not workflow and not prompt_meta:
            out["error"] = "no ComfyUI metadata"
            return out

        # Parse the prompt (API format) — flat dict of node_id -> node_config
        data = json.loads(prompt_meta) if prompt_meta else json.loads(workflow)

        for node_id, node in data.items():
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})

            if class_type == "KSampler":
                out["seed"] = inputs.get("seed", "?")
                out["steps"] = inputs.get("steps", "?")
                out["cfg"] = inputs.get("cfg", "?")
                out["sampler"] = inputs.get("sampler_name", "?")
                out["scheduler"] = inputs.get("scheduler", "?")
            elif class_type == "CLIPTextEncode":
                text = inputs.get("text", "")
                # First CLIPTextEncode = positive, second = negative (by node order)
                if "positive" not in out and not any(
                    bad in text.lower() for bad in ["blurry", "deformed", "ugly"]
                ):
                    out["positive"] = text[:80]
                elif "negative" not in out:
                    out["negative"] = text[:60]
            elif class_type == "LoraLoader":
                out["lora"] = inputs.get("lora_name", "?")
                out["lora_str"] = inputs.get("strength_model", "?")
            elif class_type == "CheckpointLoaderSimple":
                out["checkpoint"] = inputs.get("ckpt_name", "?")
    except Exception as e:
        out["error"] = str(e)
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]
    paths = glob.glob(arg) if "*" in arg else [arg]

    if not paths:
        print(f"no matches for {arg}")
        sys.exit(1)

    for p in paths:
        info = extract(p)
        print(f"\n=== {info['file']} ===")
        if "error" in info:
            print(f"  {info['error']}")
            continue
        for key in ("checkpoint", "lora", "lora_str", "seed", "steps", "cfg",
                    "sampler", "scheduler", "positive", "negative"):
            if key in info:
                print(f"  {key:12s}: {info[key]}")


if __name__ == "__main__":
    main()
