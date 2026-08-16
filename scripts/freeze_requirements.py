"""
Regenerates requirements.txt via pip freeze, prepending the PyTorch cu121
extra index URL - plain `pip freeze > requirements.txt` loses this line,
and without it `pip install -r requirements.txt` fails on a fresh machine
(the +cu121-tagged torch/torchvision wheels only exist on that index, not
on default PyPI). Confirmed this failure mode directly during Phase 2's
fresh-venv verification - always regenerate through this script, not a
bare pip freeze.
"""
import subprocess
import sys
from pathlib import Path

EXTRA_INDEX_LINE = "--extra-index-url https://download.pytorch.org/whl/cu121\n"


def main():
    req_path = Path(__file__).resolve().parent.parent / "requirements.txt"
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    with open(req_path, "w") as f:
        f.write(EXTRA_INDEX_LINE)
        f.write(result.stdout)
    print(f"Wrote {req_path}")


if __name__ == "__main__":
    main()
