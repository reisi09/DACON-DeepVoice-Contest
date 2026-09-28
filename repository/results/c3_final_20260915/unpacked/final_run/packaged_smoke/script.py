import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model/runtime"))
from run_v6_final import main
if __name__ == "__main__":
    sys.argv.insert(1, "predict")
    sys.argv.extend(["--assets", str(ROOT / "model/assets"), "--work", str(ROOT / "model/trained")])
    main()
