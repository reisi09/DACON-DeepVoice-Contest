"""Pinned AST encoder with five independently trainable DACON heads."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Final, Literal, assert_never

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import ASTConfig, ASTFeatureExtractor, ASTModel

PREDICTION_COLUMNS: Final = (
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
)
SAMPLE_RATE: Final = 16_000
WINDOW_SAMPLES: Final = 160_000
PROCESSOR_CONTRACT: Final = {
    "sampling_rate": SAMPLE_RATE,
    "num_mel_bins": 128,
    "max_length": 1024,
    "mean": -4.2677393,
    "std": 4.5689974,
}
TrainMode = Literal["frozen", "last_block"]


class ASTContractError(ValueError):
    """An offline AST asset or input violates the pinned model contract."""


def _validate_processor(processor: ASTFeatureExtractor) -> None:
    for name, expected in PROCESSOR_CONTRACT.items():
        actual = getattr(processor, name, None)
        if not isinstance(actual, int | float) or not math.isclose(float(actual), expected):
            raise ASTContractError("AST processor contract mismatch")


def load_processor(local_dir: Path) -> ASTFeatureExtractor:
    """Load the pinned processor locally and fail closed on configuration drift."""
    processor = ASTFeatureExtractor.from_pretrained(local_dir, local_files_only=True)
    _validate_processor(processor)
    return processor


def _mps_block_input_gradient(module: nn.Module, inputs: tuple) -> tuple:
    """Avoid MPS partial LayerNorm backward errors without unfreezing the prefix."""
    hidden = inputs[0]
    if (hidden.device.type == "mps" and torch.is_grad_enabled()
            and module.layernorm_before.weight.requires_grad):
        hidden.requires_grad_(True)
    return inputs


class ASTMultiTask(nn.Module):
    """AST pooler followed by five independent binary heads."""

    def __init__(self, backbone: ASTModel, mode: TrainMode) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.encoder.layer[-1].register_forward_pre_hook(_mps_block_input_gradient)
        self.heads = nn.ModuleList(nn.Linear(backbone.config.hidden_size, 1) for _ in PREDICTION_COLUMNS)
        self.mode = mode
        self.configure_trainable(mode)

    @classmethod
    def from_pretrained(cls, local_dir: Path, mode: TrainMode) -> ASTMultiTask:
        """Load only supplied local AST weights; AudioSet classifier logits are unused."""
        load_processor(local_dir)
        backbone = ASTModel.from_pretrained(
            local_dir,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        return cls(backbone, mode)

    def configure_trainable(self, mode: TrainMode) -> None:
        """Expose exactly the parameters owned by the selected training mode."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.heads.parameters():
            parameter.requires_grad_(True)
        match mode:
            case "frozen":
                pass
            case "last_block":
                for parameter in self.backbone.encoder.layer[-1].parameters():
                    parameter.requires_grad_(True)
                for parameter in self.backbone.layernorm.parameters():
                    parameter.requires_grad_(True)
            case unreachable:
                assert_never(unreachable)
        self.mode = mode

    def train(self, mode: bool = True) -> ASTMultiTask:
        """Keep frozen encoder blocks deterministic while training selected layers."""
        super().train(mode)
        if mode:
            self.backbone.eval()
            if self.mode == "last_block":
                self.backbone.encoder.layer[-1].train()
        return self

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        """Return the final-normalized CLS/distillation-token pooler output."""
        return self.backbone(input_values=input_values).pooler_output

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(input_values)
        return torch.cat([head(pooled) for head in self.heads], dim=-1)


def _eligible_mask(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim != 2 or targets.shape[1] != len(PREDICTION_COLUMNS):
        raise ASTContractError("AST targets must have five columns")
    known = torch.isfinite(targets)
    if torch.any(known & (targets != 0) & (targets != 1)):
        raise ASTContractError("AST targets must be binary or NaN")
    eligible = known.clone()
    eligible[:, 1] &= known[:, 3] & (targets[:, 3] == 1)
    eligible[:, 2] &= known[:, 4] & (targets[:, 4] == 1)
    return eligible


def compute_pos_weights(targets: torch.Tensor) -> torch.Tensor:
    """Compute bounded training-only positive weights over eligible truth."""
    eligible = _eligible_mask(targets)
    weights = torch.ones(len(PREDICTION_COLUMNS), dtype=targets.dtype, device=targets.device)
    for index in range(len(PREDICTION_COLUMNS)):
        selected = targets[:, index][eligible[:, index]]
        positives = torch.count_nonzero(selected == 1)
        negatives = torch.count_nonzero(selected == 0)
        if positives > 0 and negatives > 0:
            weights[index] = (negatives / positives).clamp(0.25, 4.0)
    return weights


def masked_multitask_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weights: torch.Tensor,
) -> torch.Tensor:
    """Mean five independently normalized BCE heads over known eligible labels."""
    if logits.shape != targets.shape or pos_weights.shape != (len(PREDICTION_COLUMNS),):
        raise ASTContractError("AST loss input contract mismatch")
    eligible = _eligible_mask(targets)
    losses: list[torch.Tensor] = []
    safe_targets = torch.nan_to_num(targets)
    for index in range(len(PREDICTION_COLUMNS)):
        mask = eligible[:, index]
        if torch.any(mask):
            losses.append(functional.binary_cross_entropy_with_logits(
                logits[:, index][mask],
                safe_targets[:, index][mask],
                pos_weight=pos_weights[index],
            ))
    if not losses:
        raise ASTContractError("AST batch has no eligible labels")
    return torch.stack(losses).mean()


def split_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_samples: int = WINDOW_SAMPLES,
) -> tuple[np.ndarray, ...]:
    """Cover one mono file with deterministic, non-overlapping windows including its tail."""
    if sample_rate != SAMPLE_RATE or window_samples <= 0 or audio.ndim != 1 or audio.size == 0:
        raise ASTContractError("AST audio input contract mismatch")
    if not np.isfinite(audio).all():
        raise ASTContractError("AST audio contains non-finite samples")
    return tuple(audio[start:start + window_samples] for start in range(0, audio.size, window_samples))


def prepare_inputs(
    processor: ASTFeatureExtractor,
    windows: tuple[np.ndarray, ...],
) -> torch.Tensor:
    """Apply the pinned AST feature extractor to one file's windows."""
    _validate_processor(processor)
    if not windows:
        raise ASTContractError("AST requires at least one audio window")
    batch = processor(list(windows), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    return batch["input_values"]


def aggregate_window_logits(logits: torch.Tensor) -> torch.Tensor:
    """Mean all window probabilities from exactly one file."""
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != len(PREDICTION_COLUMNS):
        raise ASTContractError("AST window logits contract mismatch")
    return torch.sigmoid(logits.float()).mean(dim=0)


def predict_file(
    model: ASTMultiTask,
    processor: ASTFeatureExtractor,
    audio: np.ndarray,
) -> torch.Tensor:
    """Predict one file without reading or retaining another file's state."""
    inputs = prepare_inputs(processor, split_audio(audio)).to(next(model.parameters()).device)
    with torch.inference_mode():
        return aggregate_window_logits(model(inputs)).cpu()


def export_checkpoint(
    model: ASTMultiTask,
    processor: ASTFeatureExtractor,
    directory: Path,
) -> Path:
    """Export a self-contained inference checkpoint without optimizer state."""
    _validate_processor(processor)
    directory.mkdir(parents=True, exist_ok=True)
    model.backbone.config.to_json_file(directory / "config.json")
    processor.save_pretrained(directory)
    metadata = {
        "format_version": 1,
        "mode": model.mode,
        "columns": list(PREDICTION_COLUMNS),
        "hidden_size": model.backbone.config.hidden_size,
        "processor_contract": PROCESSOR_CONTRACT,
    }
    metadata_path = directory / "multitask_config.json"
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    weights_path = directory / "model.safetensors"
    temporary_weights = weights_path.with_suffix(".safetensors.tmp")
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
        temporary_weights,
    )
    temporary_weights.replace(weights_path)
    return directory


def load_checkpoint(
    directory: Path,
    mode: TrainMode,
) -> tuple[ASTMultiTask, ASTFeatureExtractor]:
    """Load a self-contained checkpoint and reject processor/head contract drift."""
    metadata_path = directory / "multitask_config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "format_version": 1,
        "mode": mode,
        "columns": list(PREDICTION_COLUMNS),
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ASTContractError("AST checkpoint contract mismatch")
    processor = load_processor(directory)
    config = ASTConfig.from_json_file(directory / "config.json")
    if metadata.get("hidden_size") != config.hidden_size:
        raise ASTContractError("AST checkpoint contract mismatch")
    model = ASTMultiTask(ASTModel(config), mode)
    model.load_state_dict(load_file(directory / "model.safetensors"), strict=True)
    return model, processor
