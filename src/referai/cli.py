"""Interface en ligne de commande du module detection + tracking."""

import argparse
import json
import logging
import os
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
from .role_classification import (
    predict_role_tracks,
    train_role_classifier,
    validate_role_classifier,
)
from .soccernet import (
    ROLE_CLASSIFIER_ROLES,
    SOCCERNET_SPLITS,
    download_soccernet_gamestate,
    prepare_gamestate_roles,
)
from .tracking import track_video
from .tracking_evaluation import evaluate_tracking
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

    download_gs = subparsers.add_parser(
        "download-soccernet", help="Telecharge et extrait SoccerNet GameState"
    )
    download_gs.add_argument("--output", type=Path, default=Path("data/SoccerNetGS"))
    download_gs.add_argument("--task", default="gamestate-2024")
    download_gs.add_argument(
        "--split", dest="splits", action="append", choices=SOCCERNET_SPLITS
    )
    download_gs.add_argument("--no-extract", action="store_true")
    download_gs.add_argument("--delete-archives", action="store_true")
    download_gs.add_argument("--password-env", default="SOCCERNET_PASSWORD")

    prepare_roles = subparsers.add_parser(
        "prepare-gamestate-roles",
        help="Extrait les crops de roles depuis Labels-GameState.json",
    )
    prepare_roles.add_argument("--source", type=Path, required=True)
    prepare_roles.add_argument("--output", type=Path, required=True)
    prepare_roles.add_argument(
        "--split", dest="splits", action="append", choices=SOCCERNET_SPLITS
    )
    prepare_roles.add_argument(
        "--role", dest="roles", action="append", choices=ROLE_CLASSIFIER_ROLES
    )
    prepare_roles.add_argument("--frame-stride", type=int, default=5)
    prepare_roles.add_argument("--max-samples-per-track", type=int, default=40)
    prepare_roles.add_argument("--context", type=float, default=0.30)
    prepare_roles.add_argument("--min-crop-size", type=int, default=12)
    prepare_roles.add_argument("--jpeg-quality", type=int, default=95)
    prepare_roles.add_argument("--minimum-version", default="1.3")
    prepare_roles.add_argument("--max-clips", type=int)

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

    train_role = subparsers.add_parser(
        "train-role", help="Fine-tuning du classifieur de roles SoccerNet"
    )
    train_role.add_argument("--config", type=Path, required=True)
    train_role.add_argument("--hardware")
    train_role.add_argument("--resume", action="store_true")

    validate_role = subparsers.add_parser(
        "validate-role", help="Evalue le classifieur de roles image par image"
    )
    validate_role.add_argument("--weights", type=Path, required=True)
    validate_role.add_argument("--data", type=Path, required=True)
    validate_role.add_argument("--hardware")
    validate_role.add_argument("--split", choices=("val", "test"), default="val")
    validate_role.add_argument("--imgsz", type=int, default=224)
    validate_role.add_argument("--batch", type=int, default=64)

    predict_roles = subparsers.add_parser(
        "predict-role-tracks",
        help="Evalue les roles avec lissage temporel par track_id",
    )
    predict_roles.add_argument("--weights", type=Path, required=True)
    predict_roles.add_argument("--data", type=Path, required=True)
    predict_roles.add_argument("--output", type=Path, required=True)
    predict_roles.add_argument("--manifest", type=Path)
    predict_roles.add_argument("--hardware")
    predict_roles.add_argument("--split", choices=("train", "val", "test"), default="val")
    predict_roles.add_argument("--alpha", type=float, default=0.20)
    predict_roles.add_argument("--imgsz", type=int, default=224)
    predict_roles.add_argument("--batch", type=int, default=64)
    predict_roles.add_argument("--max-samples", type=int)
    predict_roles.add_argument(
        "--source-root",
        type=Path,
        help="Racine SoccerNet si le chemin source du dataset.yaml n'est plus valide",
    )
    predict_roles.add_argument("--image-prior-strength", type=float, default=0.25)
    predict_roles.add_argument("--pitch-prior-strength", type=float, default=0.75)
    predict_roles.add_argument("--spatial-bins-x", type=int, default=12)
    predict_roles.add_argument("--spatial-bins-y", type=int, default=8)
    predict_roles.add_argument("--spatial-smoothing", type=float, default=1.0)
    predict_roles.add_argument("--quality-minimum-weight", type=float, default=0.05)

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

    evaluate = subparsers.add_parser(
        "evaluate-tracking",
        help="Infere les sequences MOT puis calcule HOTA, IDF1, MOTA et AssA",
    )
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--weights", type=Path)
    evaluate.add_argument("--trackeval-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="val")
    evaluate.add_argument(
        "--tracker", type=Path, default=Path("configs/bytetrack_football.yaml")
    )
    evaluate.add_argument("--tracker-name", default="referai")
    evaluate.add_argument("--hardware")
    evaluate.add_argument("--confidence", type=float, default=0.05)
    evaluate.add_argument("--iou", type=float, default=0.70)
    evaluate.add_argument("--class-id", type=int, default=0)
    evaluate.add_argument(
        "--sequence", dest="sequences", action="append", help="Nom exact; option repetable"
    )
    evaluate.add_argument("--max-sequences", type=int)
    evaluate.add_argument("--skip-inference", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.command == "inspect-hardware":
        profile = resolve_profile(_hardware(args.hardware))
        print(profile_summary(profile, inspect_gpus(backend=profile.backend)))
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
    elif args.command == "download-soccernet":
        results = download_soccernet_gamestate(
            output=args.output,
            task=args.task,
            splits=args.splits or ("train", "valid", "test"),
            extract=not args.no_extract,
            keep_archives=not args.delete_archives,
            password=os.environ.get(args.password_env),
        )
        print(json.dumps([asdict(result) for result in results], indent=2))
    elif args.command == "prepare-gamestate-roles":
        stats = prepare_gamestate_roles(
            source=args.source,
            output=args.output,
            splits=args.splits or ("train", "valid", "test"),
            roles=args.roles or ROLE_CLASSIFIER_ROLES,
            frame_stride=args.frame_stride,
            max_samples_per_track=args.max_samples_per_track,
            context=args.context,
            min_crop_size=args.min_crop_size,
            jpeg_quality=args.jpeg_quality,
            minimum_version=args.minimum_version,
            max_clips=args.max_clips,
        )
        print(json.dumps(asdict(stats), indent=2, ensure_ascii=False))
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
    elif args.command == "train-role":
        train_role_classifier(args.config, _hardware(args.hardware), args.resume)
    elif args.command == "validate-role":
        result = validate_role_classifier(
            args.weights,
            args.data,
            _hardware(args.hardware),
            args.split,
            args.imgsz,
            args.batch,
        )
        if hasattr(result, "results_dict"):
            print(json.dumps(result.results_dict, indent=2))
    elif args.command == "predict-role-tracks":
        result = predict_role_tracks(
            weights=args.weights,
            dataset=args.data,
            output=args.output,
            split=args.split,
            manifest=args.manifest,
            hardware=_hardware(args.hardware),
            alpha=args.alpha,
            imgsz=args.imgsz,
            batch=args.batch,
            max_samples=args.max_samples,
            source_root=args.source_root,
            image_prior_strength=args.image_prior_strength,
            pitch_prior_strength=args.pitch_prior_strength,
            spatial_bins_x=args.spatial_bins_x,
            spatial_bins_y=args.spatial_bins_y,
            spatial_smoothing=args.spatial_smoothing,
            quality_minimum_weight=args.quality_minimum_weight,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
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
    elif args.command == "evaluate-tracking":
        result = evaluate_tracking(
            data_yaml=args.data,
            weights=args.weights,
            trackeval_root=args.trackeval_root,
            output=args.output,
            split=args.split,
            tracker=args.tracker,
            tracker_name=args.tracker_name,
            hardware=_hardware(args.hardware),
            confidence=args.confidence,
            iou=args.iou,
            class_id=args.class_id,
            requested_sequences=args.sequences,
            max_sequences=args.max_sequences,
            skip_inference=args.skip_inference,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
