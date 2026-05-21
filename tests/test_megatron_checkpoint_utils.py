import types

import pytest

from slime.utils.checkpoint_utils import is_megatron_checkpoint, validate_defer_optimizer_checkpoint


@pytest.mark.unit
def test_is_megatron_checkpoint_detects_tracker_file(tmp_path):
    checkpoint_dir = tmp_path / "actor"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text("1\n")

    assert is_megatron_checkpoint(checkpoint_dir)


@pytest.mark.unit
def test_is_megatron_checkpoint_detects_direct_iteration_dir(tmp_path):
    checkpoint_dir = tmp_path / "iter_0000001"
    checkpoint_dir.mkdir()

    assert is_megatron_checkpoint(checkpoint_dir)


@pytest.mark.unit
def test_is_megatron_checkpoint_rejects_hf_style_dir(tmp_path):
    checkpoint_dir = tmp_path / "hf"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}\n")

    assert not is_megatron_checkpoint(checkpoint_dir)


@pytest.mark.unit
def test_defer_optimizer_validation_requires_megatron_checkpoint():
    args = types.SimpleNamespace(defer_optimizer_init=True, load="/tmp/hf")

    with pytest.raises(ValueError, match="requires --load to resolve to a Megatron torch_dist checkpoint"):
        validate_defer_optimizer_checkpoint(args.load)


@pytest.mark.unit
def test_defer_optimizer_validation_accepts_megatron_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "actor"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text("1\n")
    args = types.SimpleNamespace(defer_optimizer_init=True, load=str(checkpoint_dir))

    validate_defer_optimizer_checkpoint(args.load)
