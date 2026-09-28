from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from redesign_assets import AssetError, prepare_assets, validate_assets
from redesign_features import extract_features, load_records
from redesign_features_cache import (
    CacheError,
    FeatureRecord,
    FeatureRow,
    config_sha256,
    extractor_sha256,
    install_receipt,
    receipt_payload,
    record_path,
    save_record,
)

PACKAGE_FILES = {
    "model/assets/baseline/script.py": b"baseline source",
    "model/assets/baseline/model/df_arena_1b/pytorch_model.bin": b"df",
    "model/assets/baseline/model/htdemucs/htdemucs.yaml": b"demucs config",
    "model/assets/baseline/model/panns/Cnn14_mAP=0.431.pth": b"panns",
    "model/assets/baseline/model/panns/component_labels.json": b"{}",
    "model/assets/mert/config.json": b"{}",
    "model/assets/mert/preprocessor_config.json": b"{}",
    "model/assets/mert/pytorch_model.bin": b"mert",
}


def _package(path: Path, extra: dict[str, bytes] | None = None) -> str:
    members = dict(PACKAGE_FILES)
    members.update(extra or {})
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr("model/manifest.json", json.dumps({"sha256": manifest}))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> FeatureRecord:
    count = 2
    voice_probability = np.asarray([0.25, 0.75], dtype=np.float32)
    presence = np.asarray([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
    aux = np.asarray([
        [-1, -2, -3, 1, 1, 1],
        [-1, -2, -3, 1, 0, 0.5],
    ], dtype=np.float32)
    return FeatureRecord(
        stem=np.zeros((count, 1024), dtype=np.float32),
        raw=np.ones((count, 1024), dtype=np.float32),
        aux=aux,
        starts=np.asarray([0, 64_600], dtype=np.int64),
        valid=np.asarray([True, True]),
        stem_audible=np.asarray([True, False]),
        voice_embedding=np.zeros((count, 1280), dtype=np.float32),
        voice_logits=np.log(voice_probability / (1 - voice_probability)).astype(np.float32),
        voice_probability=voice_probability,
        presence=presence,
        evidence=np.asarray([0.75, 0.6, 0.8], dtype=np.float32),
    )


def _cache(tmp_path: Path) -> tuple[list[FeatureRow], Path]:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fixture audio identity")
    row = FeatureRow(ID="sample", path=str(audio), audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest())
    cache = tmp_path / "cache"
    payload = {
        "extractor_sha256": extractor_sha256(),
        "model_sha256": "1" * 64,
        "source_sha256": "2" * 64,
        "config_sha256": config_sha256(),
    }
    receipt = install_receipt(cache, payload)
    target, digest = save_record(cache, row, receipt, _record())
    manifest = {
        "receipt_sha256": receipt,
        "records": {"sample": {
            "audio_sha256": row["audio_sha256"],
            "record": str(target.relative_to(cache)),
            "record_sha256": digest,
        }},
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return [row], cache


def test_prepare_assets_rejects_traversal_and_verifies_package_members(tmp_path: Path) -> None:
    malicious = tmp_path / "malicious.zip"
    digest = _package(malicious, {"model/assets/mert/../../escape": b"escape"})
    with pytest.raises(AssetError, match="Unsafe selected asset path"):
        prepare_assets(malicious, tmp_path / "bad-assets", expected_sha256=digest)

    package = tmp_path / "m330.zip"
    digest = _package(package)
    assets = tmp_path / "assets"
    hashes = prepare_assets(package, assets, expected_sha256=digest)
    assert hashes == validate_assets(assets)
    assert not (tmp_path / "escape").exists()
    (assets / "mert/config.json").write_bytes(b"tampered")
    with pytest.raises(AssetError, match="digest mismatch"):
        validate_assets(assets)


def test_load_records_rejects_changed_audio_and_semantic_cache_corruption(tmp_path: Path) -> None:
    rows, cache = _cache(tmp_path)
    loaded = load_records(rows, cache)
    np.testing.assert_allclose(loaded["sample"]["evidence"], [0.75, 0.6, 0.8], atol=1e-7, rtol=0)

    Path(rows[0]["path"]).write_bytes(b"different audio")
    with pytest.raises(CacheError, match="Audio checksum differs"):
        load_records(rows, cache)
    Path(rows[0]["path"]).write_bytes(b"fixture audio identity")

    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = manifest["receipt_sha256"]
    corrupt = _record()
    corrupt["evidence"] = np.asarray([0.1, 0.1, 0.1], dtype=np.float32)
    target = record_path(cache, "sample")
    metadata = json.dumps({"ID": "sample", "audio_sha256": rows[0]["audio_sha256"], "receipt": receipt}, sort_keys=True)
    with target.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.asarray(metadata), **corrupt)
    manifest["records"]["sample"]["record_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CacheError, match="Bag evidence"):
        load_records(rows, cache)


def test_load_records_rejects_modified_immutable_receipt(tmp_path: Path) -> None:
    rows, cache = _cache(tmp_path)
    manifest = json.loads((cache / "manifest.json").read_text())
    receipt = cache / "receipts" / f"{manifest['receipt_sha256']}.json"
    receipt.write_text(receipt.read_text() + " ", encoding="utf-8")
    with pytest.raises(CacheError, match="receipt is missing or modified"):
        load_records(rows, cache)


def test_load_records_accepts_verified_manifest_subset(tmp_path: Path) -> None:
    rows, cache = _cache(tmp_path)
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"]["another-partition-row"] = {
        "audio_sha256": "f" * 64,
        "record": "records/not-requested.npz",
        "record_sha256": "e" * 64,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert set(load_records(rows, cache)) == {"sample"}


def test_extract_features_preserves_other_partition_manifest_entries(tmp_path: Path) -> None:
    package = tmp_path / "m330.zip"
    assets = tmp_path / "assets"
    prepare_assets(package, assets, expected_sha256=_package(package))
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fixture audio identity")
    row = FeatureRow(ID="sample", path=str(audio), audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest())
    cache = tmp_path / "cache"
    receipt = install_receipt(cache, receipt_payload(assets, validate_assets(assets)))
    target, digest = save_record(cache, row, receipt, _record())
    manifest = {
        "receipt_sha256": receipt,
        "records": {
            "older": {"audio_sha256": "a" * 64, "record": "records/older.npz", "record_sha256": "b" * 64},
            "sample": {
                "audio_sha256": row["audio_sha256"], "record": str(target.relative_to(cache)),
                "record_sha256": digest,
            },
        },
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = extract_features([row], assets, cache, device="cpu")
    assert result["cached"] == 1
    assert result["windows"] == 2
    assert result["peak_rss_bytes"] > 0
    assert set(json.loads((cache / "manifest.json").read_text())["records"]) == {"older", "sample"}
