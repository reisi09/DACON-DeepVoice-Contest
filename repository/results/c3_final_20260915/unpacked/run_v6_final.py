"""Train the final pipeline or predict the competition's five probabilities."""

import argparse
import json
import time
from pathlib import Path

from v6_core import COLUMNS
from v6_data import prepare, write_json
from v6_runtime import Engine
from v6_train import train, trained_heads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "predict"))
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--work", type=Path, default=Path("v6_run"))
    parser.add_argument("--train-csv", type=Path, default=Path("v4_validation/train.csv"))
    parser.add_argument("--dev-csv", type=Path, default=Path("v4_validation/dev.csv"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-hours", type=float, default=3.)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/submission.csv"))
    args = parser.parse_args()
    if args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    args.work.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        split = prepare(args.train_csv, args.dev_csv, args.work, args.seed)
        print({k: len(split[k]) for k in ("expert", "fusion", "dev")})
        return
    engine = Engine(args.assets.resolve(), args.device)
    if args.command == "predict":
        head = trained_heads(args.work, engine.asset_id)
        base = engine.baseline
        columns, rows = base.read_sample_submission(args.sample_submission)
        files = base.order_audio_files(base.find_audio_files(args.test_dir), rows)
        for row, path in zip(rows, files, strict=True):
            if time.monotonic() >= deadline:
                raise TimeoutError("Inference deadline reached; partial submission will not be saved")
            values = head.predict(engine.cached(path, args.work / "inference_cache"))
            row.update(dict(zip(COLUMNS, values.tolist())))
        base.save_submission(args.output, columns, rows)
        return
    split = json.loads((args.work / "split.json").read_text())
    all_rows = [r for partition in ("expert", "fusion", "dev") for r in split[partition]]
    for i, row in enumerate(all_rows):
        if time.monotonic() >= deadline:
            write_json(args.work / "progress.json", dict(status="time_limit", completed=i, total=len(all_rows)))
            raise TimeoutError("Extraction deadline reached; caches retained for resume")
        engine.cached(Path(row["path"]), args.work / "cache")
        if i == 0 or (i + 1) % 25 == 0 or i + 1 == len(all_rows):
            progress = dict(status="extracting", completed=i + 1, total=len(all_rows),
                            elapsed_seconds=round(time.monotonic() - started, 1))
            write_json(args.work / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    metadata = train(split, args.work / "cache", engine.cache_id, engine.asset_id,
                     args.work, args.epochs, deadline)
    write_json(args.work / "progress.json", dict(status="complete", elapsed_seconds=time.monotonic() - started))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
