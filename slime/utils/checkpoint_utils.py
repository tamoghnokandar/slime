import re
from pathlib import Path


def is_megatron_checkpoint(path: str | Path | None) -> bool:
    if path is None:
        return False

    checkpoint_path = Path(path)
    return (checkpoint_path / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", checkpoint_path.name)
    )


def validate_defer_optimizer_checkpoint(load_path: str | Path | None) -> None:
    if not (load_path is not None and Path(load_path).exists() and is_megatron_checkpoint(load_path)):
        raise ValueError(
            "--defer-optimizer-init requires --load to resolve to a Megatron torch_dist checkpoint. "
            "HF/bridge checkpoint loading is not supported for deferred optimizer startup."
        )
