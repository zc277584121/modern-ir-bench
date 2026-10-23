"""Resolve a PEFT adapter and its backbone to immutable local snapshots."""

from __future__ import annotations

import json
from pathlib import Path


def pinned_adapter_snapshot(model: str, revision: str, base_revision: str, *, local_files_only: bool) -> Path:
    """Bind the adapter's base path without modifying either official snapshot."""
    from filelock import FileLock
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    adapter = Path(snapshot_download(model, revision=revision, local_files_only=local_files_only))
    config = json.loads((adapter / "adapter_config.json").read_text())
    base = Path(
        snapshot_download(config["base_model_name_or_path"], revision=base_revision, local_files_only=local_files_only)
    )
    target = Path(HF_HUB_CACHE) / "modern-ir-pinned-adapters" / model.replace("/", "--") / f"{revision}-{base_revision}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock"):
        target.mkdir(exist_ok=True)
        for source in adapter.rglob("*"):
            destination = target / source.relative_to(adapter)
            if source.is_dir():
                destination.mkdir(exist_ok=True)
            elif source.name != "adapter_config.json" and not destination.exists():
                destination.symlink_to(source.resolve())
        config["base_model_name_or_path"] = str(base)
        config["revision"] = base_revision
        config_path = target / "adapter_config.json"
        content = json.dumps(config, indent=2) + "\n"
        if not config_path.exists() or config_path.read_text() != content:
            config_path.write_text(content)
    return target
