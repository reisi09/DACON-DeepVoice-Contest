"""High-signal checks for residual causality, masks, matched exposure, and OOF."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from redesign_features_cache import (
    config_sha256,
    extractor_sha256,
    install_receipt,
    record_path,
    save_record,
)
from redesign_fit import fit_music, masked_bce
from redesign_file import apply_relation, fit_relation
from redesign_evaluate import open_sealed
from redesign_evaluation_common import canonical_sha256
from redesign_models import MusicResidual, StemMusicWrapper, make_control
from redesign_oof import internal_earlystop
from redesign_train import arm_rows, predict_checkpoint, predict_ensemble_artifact, run_experiments
from redesign_voice import fit_voice, predict_voice


def test_zero_residual_preserves_control_then_raw_changes_trained_output() -> None:
    # Given: identical seeded controls and a residual with distinct raw windows.
    torch.manual_seed(7)
    stem = torch.randn(4, 2, 1024)
    raw = torch.randn_like(stem)
    aux = torch.randn(4, 2, 6)
    valid = torch.ones(4, 2, dtype=torch.bool)
    control = StemMusicWrapper(make_control(41)).eval()
    residual = MusicResidual(make_control(41)).eval()

    # When: the zero residual is evaluated, then trained through two optimizer steps.
    with torch.no_grad():
        expected = control(stem, valid).bag_logits
        initial = residual(stem, raw, aux, valid).bag_logits
    assert torch.equal(expected, initial)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=0.01)
    labels = torch.tensor([0, 1, 0, 1])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(residual(stem, raw, aux, valid).bag_logits, labels).backward()
        optimizer.step()

    # Then: changing only raw order causally changes the learned prediction.
    residual.eval()
    direct = residual(stem, raw, aux, valid).bag_logits
    shuffled = residual(stem, raw.flip(0), aux, valid).bag_logits
    assert not torch.allclose(direct, shuffled, atol=1e-7, rtol=0)


def test_unknown_targets_have_zero_gradient() -> None:
    # Given: logits whose labels are entirely unknown.
    logits = torch.tensor([0.2, -0.4], requires_grad=True)
    # When: the contract loss is differentiated.
    masked_bce(logits, torch.zeros(2), torch.zeros(2, dtype=torch.bool)).backward()
    # Then: unknown labels cannot update the head.
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_pair_slots_replace_in_place_and_shortages_fail() -> None:
    # Given: two matched pair slots plus one shared observed row.
    rows = [_row("p0", "g0", 0, pair="x", view="pure", duration=4),
            _row("m0", "g0", 0, pair="x", view="mixed", duration=4),
            _row("p1", "g1", 1, pair="y", view="pure", duration=6),
            _row("m1", "g1", 1, pair="y", view="mixed", duration=6),
            _row("shared", "g2", 0, view="observed", duration=5)]
    # When: pure and mixed arms are materialized.
    pure, pure_exposure = arm_rows(rows, "A")
    mixed, mixed_exposure = arm_rows(rows, "D")
    # Then: IDs change in place while draw counts and paired seconds stay equal.
    assert {row["ID"] for row in pure} == {"p0", "p1", "shared"}
    assert {row["ID"] for row in mixed} == {"m0", "m1", "shared"}
    assert pure_exposure == mixed_exposure
    with pytest.raises(ValueError, match="Pair shortage"):
        arm_rows(rows[:-2], "D")


def test_run_experiments_refits_group_oof_and_reloads_checkpoint(tmp_path: Path) -> None:
    # Given: an on-disk labeled feature fixture with eight linked pair groups.
    data, cache = _fixture(tmp_path)
    output = tmp_path / "run"
    # When: the exact public API trains the raw/mixed arm and two honest OOF folds.
    summary = run_experiments(data, cache, output, seeds=(41,), epochs=1, patience=1,
                              batch_size=32, device="cpu", arms=("D",), folds=2)
    oof = json.loads((output / "oof/D_seed41.json").read_text())
    prediction = json.loads((output / "predictions/D_seed41.json").read_text())
    # Then: outer and internal validation groups are absent from every fold fit.
    assert summary["status"] == "complete"
    for fold in oof["folds"]:
        held, trained, early = map(set, (fold["held_out_groups"], fold["trained_groups"], fold["earlystop_groups"]))
        assert held.isdisjoint(trained | early)
        assert set(fold["train_ids"]).isdisjoint(fold["held_out_ids"])
        assert set(fold["earlystop_ids"]).isdisjoint(fold["held_out_ids"])
    assert set(oof["predictions"]) == set(oof["held_out_group"])
    checkpoint = Path(summary["arms"]["D"]["seeds"][0]["checkpoint"])
    from redesign_data import read_rows
    from redesign_features import load_records
    rows = read_rows(data / "validation.csv")
    records = load_records(read_rows(data / "train.csv") + rows, cache)
    reloaded = predict_checkpoint(checkpoint, rows, records)
    for identifier, values in prediction["predictions"].items():
        assert reloaded[identifier]["MUSIC_FAKE_PROB"] == pytest.approx(values["MUSIC_FAKE_PROB"], abs=1e-7)


def test_checkpoint_identity_rejects_relabel_and_changed_features(tmp_path: Path) -> None:
    # Given: a completed checkpoint bound to labels, groups, audio, features, source, and recipe.
    data, cache = _fixture(tmp_path)
    from redesign_data import read_rows
    from redesign_features import load_records
    rows = read_rows(data / "train.csv")
    records = load_records(rows + read_rows(data / "validation.csv"), cache)
    selected, _ = arm_rows(rows, "D")
    fitting, early = internal_earlystop(selected, 41)
    output = tmp_path / "identity"
    fit_music(fitting, records, early, output, arm="D", seed=41, epochs=1, patience=1,
              batch_size=32, device="cpu", deadline=None)

    # When/Then: the same IDs with a changed truth label cannot reuse that checkpoint.
    relabelled = [dict(row) for row in fitting]
    relabelled[0]["MUSIC_FAKE"] = str(1 - int(relabelled[0]["MUSIC_FAKE"]))
    with pytest.raises(ValueError, match="identity mismatch"):
        fit_music(relabelled, records, early, output, arm="D", seed=41, epochs=1, patience=1,
                  batch_size=32, device="cpu", deadline=None)

    # When/Then: changed feature content under the same IDs is also rejected.
    changed = {identifier: {name: value.copy() for name, value in record.items()}
               for identifier, record in records.items()}
    changed[fitting[0]["ID"]]["raw"][0, 0] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        fit_music(fitting, changed, early, output, arm="D", seed=41, epochs=1, patience=1,
                  batch_size=32, device="cpu", deadline=None)


def test_eligible_voice_candidate_changes_voice_and_file_and_reloads(tmp_path: Path) -> None:
    # Given: complete known real/fake singing and telephone event truth.
    data, cache = _fixture(tmp_path, with_voice=True)
    output = tmp_path / "voice-run"
    # When: the public runner fits both the music candidate and eligible voice residual.
    summary = run_experiments(data, cache, output, seeds=(41, 42), epochs=1, patience=1,
                              batch_size=32, device="cpu", arms=("D",), folds=2)
    base = json.loads((output / "predictions/D_seed41.json").read_text())
    voice = json.loads((output / "predictions/D_voice_seed41.json").read_text())
    voice_ensemble = json.loads((output / "predictions/D_voice_ensemble.json").read_text())
    voice_oof = json.loads((output / "oof/D_voice_seed41.json").read_text())
    # Then: the trained branch changes both its output and File fusion, and OOF matches its family.
    assert summary["voice"]["status"] == "complete"
    assert any(base["predictions"][key]["VOICE_FAKE_PROB"] !=
               voice["predictions"][key]["VOICE_FAKE_PROB"] for key in base["predictions"])
    assert any(base["predictions"][key]["FILE_FAKE_PROB"] !=
               voice["predictions"][key]["FILE_FAKE_PROB"] for key in base["predictions"])
    assert voice["upstream_identity"]["upstream_family_identity"] == \
        voice_oof["upstream_identity"]["upstream_family_identity"]
    checkpoint = Path(summary["voice"]["members"][0]["checkpoint"])
    from redesign_data import read_rows
    from redesign_features import load_records
    validation = read_rows(data / "validation.csv")
    training = read_rows(data / "train.csv")
    records = load_records(training + validation, cache)
    first = predict_voice(checkpoint, validation, records)
    second = predict_voice(checkpoint, validation, records)
    assert first == second
    repeated = fit_voice(training, records, tmp_path / "voice-repeat", seed=41, epochs=1,
                         patience=1, device="cpu", deadline=None)
    assert repeated.checkpoint is not None
    assert predict_voice(repeated.checkpoint, validation, records) == first
    members = summary["voice"]["members"]
    recreated = predict_ensemble_artifact(
        [Path(member["music_checkpoint"]) for member in members], validation, records,
        training_manifest_sha256=voice["upstream_identity"]["training_manifest_sha256"],
        records_identity=voice["upstream_identity"]["records_identity"],
        candidate="D_voice_ensemble",
        voice_checkpoints=[Path(member["checkpoint"]) for member in members],
    )
    assert recreated["predictions"] == voice_ensemble["predictions"]
    assert recreated["upstream_identity"] == voice_ensemble["upstream_identity"]
    assert recreated["upstream_identity"]["seed_voice_checkpoint_sha256"] == [
        hashlib.sha256(Path(member["checkpoint"]).read_bytes()).hexdigest() for member in members
    ]

    # Changing only seed 2 voice weights must alter the full identity and fail the old sealed lock.
    changed_checkpoint = tmp_path / "changed-voice.pt"
    payload = torch.load(Path(members[1]["checkpoint"]), map_location="cpu", weights_only=True)
    first_name = next(iter(payload["state_dict"]))
    payload["state_dict"][first_name].view(-1)[0] += 1
    torch.save(payload, changed_checkpoint)
    changed = predict_ensemble_artifact(
        [Path(member["music_checkpoint"]) for member in members], validation, records,
        training_manifest_sha256=voice["upstream_identity"]["training_manifest_sha256"],
        records_identity=voice["upstream_identity"]["records_identity"],
        candidate="D_voice_ensemble",
        voice_checkpoints=[Path(members[0]["checkpoint"]), changed_checkpoint],
    )
    assert changed["upstream_identity"] != recreated["upstream_identity"]
    assert changed["upstream_identity"]["upstream_family_identity"] == \
        recreated["upstream_identity"]["upstream_family_identity"]
    assert changed["upstream_identity"]["seed_voice_checkpoint_sha256"][0] == \
        recreated["upstream_identity"]["seed_voice_checkpoint_sha256"][0]
    assert changed["upstream_identity"]["seed_voice_checkpoint_sha256"][1] != \
        recreated["upstream_identity"]["seed_voice_checkpoint_sha256"][1]
    panel_identity = "a" * 64
    lock_payload = {"schema_version": 1, "candidate": "D_voice_ensemble",
                    "prediction_sha256": "fixture", "report_sha256": "fixture",
                    "model_identity_sha256": canonical_sha256(recreated["upstream_identity"]),
                    "sealed_panel_identity": panel_identity,
                    "report_status": "provisional"}
    lock_payload["lock_sha256"] = canonical_sha256(lock_payload)
    lock_path, changed_path = tmp_path / "lock.json", tmp_path / "changed.json"
    lock_path.write_text(json.dumps(lock_payload))
    changed_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="model identity"):
        open_sealed(lock_path, changed_path, tmp_path / "must-not-open.json",
                    panel_identity=panel_identity, access_registry=tmp_path / "registry")


def test_two_seed_oof_members_fit_and_apply_relation(tmp_path: Path) -> None:
    # Given: two independently initialized seed members on the same matched exposure.
    data, cache = _fixture(tmp_path)
    output = tmp_path / "two-seed"
    run_experiments(data, cache, output, seeds=(41, 42), epochs=1, patience=1,
                    batch_size=32, device="cpu", arms=("D",), folds=2)
    oof_path = output / "oof/D_ensemble.json"
    prediction_path = output / "predictions/D_ensemble.json"
    oof = json.loads(oof_path.read_text())

    # When: the real relation fitter validates every member and fits the ensemble OOF.
    relation_path = output / "relation.json"
    applied_path = output / "applied.json"
    fit_relation(oof_path, relation_path)
    applied = apply_relation(prediction_path, relation_path, applied_path)

    # Then: both full member proofs remain and the ensemble is their exact arithmetic mean.
    assert len(oof["oof_members"]) == 2
    identifier = next(iter(oof["predictions"]))
    expected = np.mean([member["predictions"][identifier]["MUSIC_FAKE_PROB"]
                        for member in oof["oof_members"]])
    assert oof["predictions"][identifier]["MUSIC_FAKE_PROB"] == pytest.approx(expected, abs=1e-12)
    assert applied["file_relation"]["upstream_family_identity"] == \
        oof["upstream_identity"]["upstream_family_identity"]


def _row(
    identifier: str, group: str, label: int, *, pair: str = "", view: str = "observed", duration: float = 8.075
) -> dict[str, str]:
    event = [{"component": "music", "start": 0, "end": duration, "label": label}]
    return {"ID": identifier, "path": "", "audio_sha256": "", "source": "fixture",
            "license": "fixture", "license_url": "fixture", "source_url": "fixture",
            "parent_ids": "[]", "artist_id": "", "speaker_id": "", "generator": "",
            "kind": "instrumental", "domain": "natural", "FILE_FAKE": str(label),
            "VOICE_FAKE": "", "MUSIC_FAKE": str(label), "VOICE_PRESENT": "0",
            "MUSIC_PRESENT": "1", "group_id": group, "split": "train", "pair_id": pair,
            "view": view, "event_intervals": json.dumps(event),
            "duration_seconds": str(duration)}


def _fixture(root: Path, *, with_voice: bool = False) -> tuple[Path, Path]:
    data, cache = root / "data", root / "cache"
    data.mkdir()
    audio = root / "fixture.wav"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16000)
    audio_sha = hashlib.sha256(audio.read_bytes()).hexdigest()
    train = []
    for index in range(8):
        group = f"music{index}|voice{index}"
        for view in ("pure", "mixed"):
            row = _row(f"{view}{index}", group, index % 2, pair=f"pair{index}", view=view)
            row.update(path=str(audio), audio_sha256=audio_sha)
            train.append(row)
    if with_voice:
        for domain, kind in (("natural", "singing"), ("telephone", "speech")):
            for label in (0, 1):
                row = _row(f"{domain}-{label}", f"{domain}-group-{label}", 0)
                row.update(kind=kind, domain=domain, VOICE_PRESENT="1", VOICE_FAKE=str(label),
                           FILE_FAKE=str(label), MUSIC_PRESENT="1" if kind == "singing" else "0",
                           MUSIC_FAKE="0" if kind == "singing" else "",
                           event_intervals=json.dumps([
                               {"component": "voice", "start": 0, "end": 8.075, "label": label},
                               *([{"component": "music", "start": 0, "end": 8.075, "label": 0}]
                                 if kind == "singing" else []),
                           ]))
                row.update(path=str(audio), audio_sha256=audio_sha)
                train.append(row)
    validation = []
    for index in range(2):
        row = _row(f"validation{index}", f"validation-group{index}", index)
        if with_voice:
            row.update(kind="singing", VOICE_PRESENT="1", VOICE_FAKE=str(index),
                       event_intervals=json.dumps([
                           {"component": "voice", "start": 0, "end": 8.075, "label": index},
                           {"component": "music", "start": 0, "end": 8.075, "label": index},
                       ]))
        row.update(path=str(audio), audio_sha256=audio_sha, split="validation")
        validation.append(row)
    fields = list(train[0])
    for name, rows in (("train.csv", train), ("validation.csv", validation)):
        with (data / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    payload = {"extractor_sha256": extractor_sha256(), "model_sha256": "fixture",
               "source_sha256": "fixture", "config_sha256": config_sha256()}
    receipt = install_receipt(cache, payload)
    entries = {}
    rng = np.random.default_rng(7)
    for row in train + validation:
        label = float(row["MUSIC_FAKE"] or 0)
        stem = rng.normal(label * 0.2, 0.1, (2, 1024)).astype(np.float32)
        raw = rng.normal(label * 0.8, 0.1, (2, 1024)).astype(np.float32)
        aux = np.ones((2, 6), dtype=np.float32)
        voice_label = float(row["VOICE_FAKE"]) if row["VOICE_FAKE"] else 0.0
        voice_present = row["VOICE_PRESENT"] == "1"
        presence = np.array([[0.9 if voice_present else 0.0, 0.2 if voice_present else 1.0]] * 2,
                            dtype=np.float32)
        voice_probability = np.full(2, 0.5, dtype=np.float32)
        values = {"stem": stem, "raw": raw, "aux": aux,
                  "starts": np.array([0, 64600], dtype=np.int64),
                  "valid": np.ones(2, dtype=bool), "stem_audible": np.ones(2, dtype=bool),
                  "voice_embedding": np.full((2, 1280), voice_label, dtype=np.float32),
                  "voice_logits": np.zeros(2, dtype=np.float32),
                  "voice_probability": voice_probability,
                  "presence": presence,
                  "evidence": np.array([0.5, presence[:, 0].max(), presence[:, 1].max()], dtype=np.float32)}
        target, digest = save_record(cache, row, receipt, values)
        entries[row["ID"]] = {"audio_sha256": audio_sha,
                               "record": str(target.relative_to(cache)), "record_sha256": digest}
    (cache / "manifest.json").write_text(json.dumps({"receipt_sha256": receipt, "records": entries}))
    return data, cache
