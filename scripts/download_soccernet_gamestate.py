"""Download SoccerNet GameState archives with the official SoccerNet SDK."""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
SOCCERNET_SPLITS = ("train", "valid", "test", "challenge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/SoccerNetGS"))
    parser.add_argument("--task", default="gamestate-2024")
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=SOCCERNET_SPLITS,
        help="Option repetable; defaut: train, valid, test",
    )
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="Supprime les zip uniquement apres une extraction reussie",
    )
    parser.add_argument(
        "--password-env",
        default="SOCCERNET_PASSWORD",
        help="Variable d'environnement contenant le mot de passe NDA eventuel",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from referai.soccernet import download_soccernet_gamestate

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    results = download_soccernet_gamestate(
        output=args.output,
        task=args.task,
        splits=args.splits or ("train", "valid", "test"),
        extract=not args.no_extract,
        keep_archives=not args.delete_archives,
        password=os.environ.get(args.password_env),
    )
    print(json.dumps([asdict(result) for result in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
