import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'model/runtime'))
from full_bank_submission import run
if __name__=='__main__':
    run(ROOT)
