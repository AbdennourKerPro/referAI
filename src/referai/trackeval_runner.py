"""Lance TrackEval avec les alias NumPy retires avant NumPy 1.26."""

import runpy
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m referai.trackeval_runner <script TrackEval> [arguments]")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy est requis par TrackEval") from exc
    for name, value in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    script = Path(sys.argv[1]).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError("Script TrackEval introuvable: {}".format(script))
    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
