"""Regression checks for finite-budget parent exposure and consistency training."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from eight_heads import FrozenHead, balanced_draws, consistency_draws, fit_head


def test_source_generator_rotation_reaches_every_parent_throughout_training() -> None:
    # Given: even parent counts and unequal generator counts reproduce ordinal aliasing.
    rows = [{"ID": f"{label}_{source}_{generator}_{parent}_{view}",
             "parent_id": f"{label}_{source}_{generator}_{parent}", "source": source,
             "generator": generator, "label": label}
            for label in (0, 1) for source in ("A", "B")
            for generator in (("",) if not label else (("G1", "G2") if source == "A" else ("G1", "G2", "G3")))
            for parent in range(6) for view in range(1 + 4 * (parent == 0))]

    # When: a fixed, reproducible 500-update schedule runs without a coverage prefix.
    draws = balanced_draws(rows, steps=500, batch_size=32, seed=41, mode="source_generator_parent")

    # Then: sources and generators retain their weights; no parent vanishes in later updates.
    for batch in draws:
        assert Counter(rows[index]["label"] for index in batch) == {0: 16, 1: 16}
    for start in (0, 250):
        counts = Counter(rows[index]["parent_id"] for batch in draws[start:start + 250] for index in batch)
        assert set(counts) == {row["parent_id"] for row in rows}
        for label in (0, 1):
            for source in ("A", "B"):
                eligible = [row for row in rows if row["label"] == label and row["source"] == source]
                generators = {row["generator"] for row in eligible}
                totals = [sum(counts[parent] for parent in {row["parent_id"] for row in eligible
                                                            if row["generator"] == generator})
                          for generator in generators]
                assert sum(totals) == 2000
                assert max(totals) - min(totals) <= 1
                for generator in generators:
                    parent_counts = [counts[parent] for parent in {row["parent_id"] for row in eligible
                                                                  if row["generator"] == generator}]
                    assert max(parent_counts) - min(parent_counts) <= 1
    assert draws == balanced_draws(rows, steps=500, batch_size=32, seed=41, mode="source_generator_parent")
    assert draws != balanced_draws(rows, steps=500, batch_size=32, seed=42, mode="source_generator_parent")


def test_variant_rotation_does_not_alias_parent_rotation() -> None:
    # Given: three variants and six parents, so shared ordinal modulo only reaches two parents per variant.
    rows = [{"ID": f"{label}_{parent}_{variant}", "parent_id": f"{label}_{parent}",
             "variant": variant, "label": label}
            for label in (0, 1) for parent in range(6) for variant in ("base", "telephone", "mp3_32k")]

    # When: each variant is repeatedly selected by the real sampler.
    draws = balanced_draws(rows, steps=30, batch_size=4, seed=41, mode="variant_parent")

    # Then: every parent/view cell is reached and uniformly revisited, not just every parent globally.
    counts = Counter(index for batch in draws for index in batch)
    for label in (0, 1):
        for variant in ("base", "telephone", "mp3_32k"):
            values = [counts[index] for index, row in enumerate(rows)
                      if row["label"] == label and row["variant"] == variant]
            assert min(values) > 0
            assert max(values) - min(values) <= 1
            assert sum(values) == 20


def test_fit_head_consistency_spreads_a_fixed_budget_across_classes_and_parents(tmp_path: Path) -> None:
    # Given: the historical E3 class sizes, manifest-ordered codec pairs, and distinct row feature IDs.
    rows = [{"ID": f"{label}_{parent}_{variant}", "parent_id": f"{label}_{parent}",
             "variant": variant, "label": label, "source": "S", "generator": "G" if label else ""}
            for label, count in ((0, 139), (1, 692)) for parent in range(count)
            for variant in ("base", "telephone", "mp3_32k")]
    pairs = [(index, index + offset) for index in range(0, len(rows), 3) for offset in (1, 2)]
    features = np.arange(len(rows), dtype=np.float32)[:, None] / len(rows)
    seen: list[int] = []

    def record_inputs(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if isinstance(module, FrozenHead) and len(inputs[0]) == 1:
            seen.append(round(float(inputs[0][0, 0]) * len(rows)))

    # When: real CE and consistency forwards/backwards execute for the original 500 updates.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    hook = torch.nn.modules.module.register_module_forward_pre_hook(record_inputs)
    try:
        result = fit_head("consistency", rows, features, tmp_path, steps=500, batch_size=4,
                          seed=41, mode="variant_parent", paired_indices=pairs, consistency_weight=0.1)
    finally:
        hook.remove()
        torch.set_num_threads(threads)

    # Then: exactly one genuine pair is optimized per update, with balanced finite parent exposure.
    used = list(zip(seen[::2], seen[1::2], strict=True))
    assert len(used) == 500
    assert set(used).issubset(pairs)
    assert Counter(rows[first]["label"] for first, _ in used) == {0: 250, 1: 250}
    for label, unique in ((0, 139), (1, 250)):
        counts = Counter(rows[first]["parent_id"] for first, _ in used if rows[first]["label"] == label)
        assert len(counts) == unique
        assert max(counts.values()) - min(counts.values()) <= 1
    assert len(set(used)) == 500
    assert used == consistency_draws(rows, pairs, steps=500, seed=41)
    assert used != consistency_draws(rows, pairs, steps=500, seed=42)
    assert set(consistency_draws(rows, pairs, steps=2768, seed=41)) == set(pairs)
    assert all(torch.isfinite(parameter).all() for parameter in result.model.parameters())
