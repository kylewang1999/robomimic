"""Canonical utilities for behavior-anchored frozen observation encoders."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn


OBS_ENCODER_PREFIX = "policy.obs_encoder."


@dataclass(frozen=True)
class FrozenEncoderAnchor:
    selector_path: Path
    resolved_checkpoint_path: Path
    checkpoint_sha256: str
    active_encoder_sha256: str
    encoder_state: dict[str, torch.Tensor]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative_path(path: Path, repo_root: Path) -> str:
    repo_root = Path(repo_root).resolve(strict=True)
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.absolute().relative_to(repo_root).as_posix()


def resolve_selector(
    selector: Path,
    *,
    repo_root: Path,
    expected_resolved_checkpoint: Path | None = None,
) -> tuple[Path, Path]:
    repo_root = Path(repo_root).resolve(strict=True)
    selector_path = Path(selector).expanduser()
    if not selector_path.is_absolute():
        selector_path = repo_root / selector_path
    selector_path = selector_path.absolute()

    if not selector_path.is_symlink():
        if not selector_path.exists():
            raise FileNotFoundError(f"Anchor selector not found: {selector_path}")
        raise ValueError(f"Anchor selector must be a symlink: {selector_path}")

    try:
        resolved = selector_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Anchor selector is broken: {selector_path}") from exc
    if not resolved.is_file():
        raise ValueError(f"Anchor selector must resolve to a file: {resolved}")

    repository_relative_path(selector_path, repo_root)
    repository_relative_path(resolved, repo_root)
    if expected_resolved_checkpoint is not None:
        expected = Path(expected_resolved_checkpoint).expanduser()
        if not expected.is_absolute():
            expected = repo_root / expected
        expected = expected.resolve(strict=True)
        if resolved != expected:
            raise ValueError(
                "Anchor selector resolved-path mismatch: "
                f"expected {expected}, got {resolved}"
            )
    return selector_path, resolved


def extract_obs_encoder_state(
    checkpoint: Mapping[str, object],
    *,
    source: str = "ema",
) -> dict[str, torch.Tensor]:
    if source != "ema":
        raise ValueError(f"Frozen observation encoder source must be 'ema', got {source!r}")
    model = checkpoint.get("model")
    if not isinstance(model, Mapping):
        raise KeyError("Anchor checkpoint is missing mapping checkpoint['model']")
    state = model.get(source)
    if not isinstance(state, Mapping):
        raise KeyError(f"Anchor checkpoint is missing mapping checkpoint['model']['{source}']")

    return extract_prefixed_obs_encoder_state(state)


def extract_prefixed_obs_encoder_state(
    state: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    encoder_state = {
        str(key)[len(OBS_ENCODER_PREFIX) :]: value
        for key, value in state.items()
        if str(key).startswith(OBS_ENCODER_PREFIX)
    }
    if not encoder_state:
        raise KeyError(
            f"Checkpoint state has no keys below {OBS_ENCODER_PREFIX!r}"
        )
    if not all(isinstance(value, torch.Tensor) for value in encoder_state.values()):
        raise TypeError("Observation encoder state must contain only tensors")
    return encoder_state


def encoder_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, tensor in state.items():
        key = str(raw_key)
        if key.startswith(OBS_ENCODER_PREFIX):
            key = key[len(OBS_ENCODER_PREFIX) :]
        if key in normalized:
            raise ValueError(f"Duplicate normalized encoder key: {key}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Encoder state value for {key!r} is not a tensor")
        normalized[key] = tensor

    digest = hashlib.sha256()
    for key in sorted(normalized):
        tensor = normalized[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def assert_encoder_state_equal(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> None:
    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        raise ValueError(
            "Observation encoder key mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    for key in sorted(expected):
        actual_tensor = actual[key].detach().cpu()
        expected_tensor = expected[key].detach().cpu()
        if actual_tensor.dtype != expected_tensor.dtype:
            raise ValueError(
                f"Observation encoder dtype mismatch for {key}: "
                f"{actual_tensor.dtype} != {expected_tensor.dtype}"
            )
        if actual_tensor.shape != expected_tensor.shape:
            raise ValueError(
                f"Observation encoder shape mismatch for {key}: "
                f"{tuple(actual_tensor.shape)} != {tuple(expected_tensor.shape)}"
            )
        if not torch.equal(actual_tensor, expected_tensor):
            raise ValueError(f"Observation encoder tensor mismatch for {key}")


def copy_encoder_state(
    module: nn.Module,
    encoder_state: Mapping[str, torch.Tensor],
) -> None:
    module.load_state_dict(encoder_state, strict=True)
    assert_encoder_state_equal(module.state_dict(), encoder_state)


def load_frozen_encoder_anchor(
    selector: Path,
    *,
    repo_root: Path,
    expected_checkpoint_sha256: str,
    expected_active_encoder_sha256: str,
    source: str = "ema",
    expected_resolved_checkpoint: Path | None = None,
) -> FrozenEncoderAnchor:
    selector_path, resolved = resolve_selector(
        selector,
        repo_root=repo_root,
        expected_resolved_checkpoint=expected_resolved_checkpoint,
    )
    checkpoint_sha256 = file_sha256(resolved)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "Anchor checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_sha256}, got {checkpoint_sha256}"
        )

    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Anchor checkpoint must deserialize to a mapping: {resolved}")
    encoder_state = extract_obs_encoder_state(checkpoint, source=source)
    active_encoder_sha256 = encoder_state_sha256(encoder_state)
    if active_encoder_sha256 != expected_active_encoder_sha256:
        raise ValueError(
            "Active observation encoder SHA-256 mismatch: "
            f"expected {expected_active_encoder_sha256}, got {active_encoder_sha256}"
        )

    return FrozenEncoderAnchor(
        selector_path=selector_path,
        resolved_checkpoint_path=resolved,
        checkpoint_sha256=checkpoint_sha256,
        active_encoder_sha256=active_encoder_sha256,
        encoder_state=encoder_state,
    )
