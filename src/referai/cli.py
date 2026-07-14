"""Interface en ligne de commande du module detection + tracking."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from .augment import augment_training_dataset
from .config import load_yaml
from .data import (
    create_match_map_template,
    create_oversampled_dataset,
    parse_class_map,
    prepare_mot_dataset,
)
from .hardware import inspect_gpus, profile_summary, resolve_profile
from .tracking import track_video
from .training import train_detector, validate_detector
from .visualization import visualize_mot_sequences


def _hardware(path: Optional[str]) -> Dict[str, Any]:
    return load_yaml(Path(path)) if path else {}


def _ratios(value: str) -> Tuple[float, float, float]:
    parts = tuple(float(item) for item in value.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Format attendu: train,val,test (ex. 0.7,0.15,0.15)")
    return parts  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="referai-football")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hardware = subparsers.add_parser("inspect-hardware", help="Affiche le profil de calcul resolu")
    hardware.add_argument("--hardware", help="YAML materiel optionnel")

    prepare = subparsers.add_parser("prepare-mot", help="Convertit SoccerNet/SportsMOT vers YOLO")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--split-strategy", choices=("existing", "by-match"), default="existing")
    prepare.add_argument("--match-map", type=Path)
    prepare.add_argument("--sequence-list", type=Path)
    prepare.add_argument("--class-map", type=Path)
    prepare.add_argument("--ratios", type=_ratios, default=(0.70, 0.15, 0.15))
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")

    visualize = subparsers.add_parser(
        "visualize-mot", help="Cree des videos de sequences MOT avec leur verite terrain"
    )
    visualize.add_argument("--source", type=Path, required=True)
    visualize.add_argument("--output", type=Path, required=True)
    visualize.add_argument("-n", "--num-sequences", type=int, default=2)
    visualize.add_argument("--sequence-list", type=Path)
    visualize.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    visualize.add_argument(
        "--sequence", dest="sequences", action="append", help="Nom exact; option repetable"
    )
    boxes = visualize.add_mutually_exclusive_group()
    boxes.add_argument("--boxes", dest="show_boxes", action="store_true")
    boxes.add_argument("--no-boxes", dest="show_boxes", action="store_false")
    visualize.set_defaults(show_boxes=True)
    visualize.add_argument("--shuffle", action="store_true")
    visualize.add_argument("--seed", type=int, default=42)
    visualize.add_argument("--max-frames", type=int)
    visualize.add_argument("--fps", type=float, help="Remplace le FPS de seqinfo.ini")

    match_map = subparsers.add_parser(
        "create-match-map", help="Cree le CSV sequence -> match a completer"
    )
    match_map.add_argument("--source", type=Path, required=True)
    match_map.add_argument("--output", type=Path, required=True)
    match_map.add_argument("--sequence-list", type=Path)
    match_map.add_argument("--force", action="store_true")

    oversample = subparsers.add_parser(
        "oversample", help="Surechantillonne les images d'une classe rare"
    )
    oversample.add_argument("--data", type=Path, required=True)
    oversample.add_argument("--class-id", type=int, default=3)
    oversample.add_argument("--factor", type=int, default=4)
    oversample.add_argument("--output", type=Path)

    augment = subparsers.add_parser(
        "augment", help="Genere flou, compression, sous-resolution et occultations"
    )
    augment.add_argument("--data", type=Path, required=True)
    augment.add_argument("--copies", type=int, default=1)
    augment.add_argument("--seed", type=int, default=42)
    augment.add_argument("--max-images", type=int)

    train = subparsers.add_parser("train", help="Fine-tuning YOLO resilient au materiel")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--hardware", help="YAML materiel surchargeant le fichier d'entrainement")
    train.add_argument("--resume", action="store_true")

    validate = subparsers.add_parser("validate", help="Evalue le detecteur")
    validate.add_argument("--weights", type=Path, required=True)
    validate.add_argument("--data", type=Path, required=True)
    validate.add_argument("--hardware")
    validate.add_argument("--split", choices=("val", "test"), default="val")

    track = subparsers.add_parser("track", help="Detecte et suit les objets d'une video")
    track.add_argument("--video", type=Path, required=True)
    track.add_argument("--weights", type=Path, required=True)
    track.add_argument("--output", type=Path, required=True)
    track.add_argument("--tracker", type=Path, default=Path("configs/bytetrack_football.yaml"))
    track.add_argument("--hardware")
    track.add_argument("--annotated-video", type=Path)
    track.add_argument("--mot-output", type=Path)
    track.add_argument("--trajectories-output", type=Path)
    track.add_argument("--confidence", type=float, default=0.05)
    track.add_argument("--iou", type=float, default=0.70)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.command == "inspect-hardware":
        profile = resolve_profile(_hardware(args.hardware))
        print(profile_summary(profile, inspect_gpus()))
    elif args.command == "prepare-mot":
        stats = prepare_mot_dataset(
            source=args.source,
            output=args.output,
            split_strategy=args.split_strategy,
            match_map=args.match_map,
            sequence_list=args.sequence_list,
            class_map=parse_class_map(args.class_map),
            ratios=args.ratios,
            seed=args.seed,
            link_mode=args.link_mode,
        )
        print(json.dumps(asdict(stats), indent=2))
    elif args.command == "visualize-mot":
        results = visualize_mot_sequences(
            source=args.source,
            output=args.output,
            num_sequences=args.num_sequences,
            sequence_list=args.sequence_list,
            split=args.split,
            sequence_names=args.sequences,
            show_boxes=args.show_boxes,
            shuffle=args.shuffle,
            seed=args.seed,
            max_frames=args.max_frames,
            fps=args.fps,
        )
        print(json.dumps([result.to_dict() for result in results], indent=2))
    elif args.command == "create-match-map":
        count = create_match_map_template(
            args.source, args.output, args.sequence_list, args.force
        )
        print("{} sequence(s) ecrites dans {}".format(count, args.output.resolve()))
    elif args.command == "oversample":
        print(create_oversampled_dataset(args.data, args.class_id, args.factor, args.output))
    elif args.command == "augment":
        stats = augment_training_dataset(args.data, args.copies, args.seed, args.max_images)
        print(json.dumps(asdict(stats), indent=2))
    elif args.command == "train":
        train_detector(args.config, _hardware(args.hardware), args.resume)
    elif args.command == "validate":
        result = validate_detector(args.weights, args.data, _hardware(args.hardware), args.split)
        if hasattr(result, "results_dict"):
            print(json.dumps(result.results_dict, indent=2))
    elif args.command == "track":
        stats = track_video(
            video=args.video,
            weights=args.weights,
            output=args.output,
            tracker=args.tracker,
            hardware=_hardware(args.hardware),
            annotated_video=args.annotated_video,
            mot_output=args.mot_output,
            trajectories_output=args.trajectories_output,
            confidence=args.confidence,
            iou=args.iou,
        )
        print(json.dumps(stats.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
