"""Training and temporally-smoothed inference for SoccerNet person roles."""

import gc
import json
import logging
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import load_yaml, resolve_relative_paths
from .hardware import (
    empty_accelerator_cache,
    inspect_gpus,
    is_accelerator_oom,
    profile_summary,
    resolve_profile,
)
from .role_postprocessing import (
    fit_spatial_prior,
    fuse_with_spatial_prior,
    normalized_image_position,
    normalized_pitch_position,
    observation_quality,
    spatial_prior_coordinates,
    weighted_mean_probabilities,
)
from .soccernet import ROLE_CLASSIFIER_ROLES, read_jsonl
from .training import _import_yolo

LOGGER = logging.getLogger(__name__)


def balanced_class_sample_weights(labels: Iterable[int]) -> List[float]:
    """Assign inverse-frequency sampling weights without copying rare crops."""
    values = [int(label) for label in labels]
    if not values:
        raise ValueError("Le dataset de roles ne contient aucun crop")
    counts: DefaultDict[int, int] = defaultdict(int)
    for label in values:
        counts[label] += 1
    return [1.0 / counts[label] for label in values]


def macro_f1_from_indices(
    truth: Sequence[int], predictions: Sequence[int], class_count: int
) -> float:
    """Compute macro-F1 over observed classes without a scikit-learn dependency."""
    if len(truth) != len(predictions):
        raise ValueError("truth et predictions doivent avoir la meme longueur")
    if class_count < 1:
        raise ValueError("class_count doit etre >= 1")
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for actual, predicted in zip(truth, predictions):
        if 0 <= actual < class_count and 0 <= predicted < class_count:
            confusion[actual][predicted] += 1
    f1_values = []
    for index in range(class_count):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        false_positive = sum(confusion[other][index] for other in range(class_count)) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        if support:
            f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1_values) / len(f1_values) if f1_values else 0.0


def mean_probabilities(total: Sequence[float], count: int) -> List[float]:
    """Return a normalized mean probability vector accumulated over one track."""
    if count < 1:
        raise ValueError("count doit etre >= 1")
    values = [max(0.0, float(value) / count) for value in total]
    normalizer = sum(values)
    if normalizer <= 0:
        raise ValueError("Probabilites agregees invalides")
    return [value / normalizer for value in values]


def _disable_optimizer_foreach(optimizer: Any) -> Any:
    """Avoid unstable multi-tensor optimizer kernels on Windows XPU."""
    optimizer.defaults["foreach"] = False
    for group in optimizer.param_groups:
        group["foreach"] = False
    return optimizer


def _role_classification_trainer(backend: str) -> Any:
    """Build an Ultralytics trainer balanced by role and selected on macro-F1."""
    from copy import copy

    import torch
    from torch.utils.data import Sampler, WeightedRandomSampler

    from ultralytics.data.build import InfiniteDataLoader, build_dataloader, seed_worker
    from ultralytics.data.utils import PIN_MEMORY
    from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
    from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER
    from ultralytics.utils.metrics import ClassifyMetrics
    from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first

    class MacroF1Metrics(ClassifyMetrics):
        """Classification metrics whose fitness is macro-F1, not top-1 accuracy."""

        def __init__(self) -> None:
            super().__init__()
            self.macro_f1 = 0.0
            self.class_count = 0

        def process(self, targets: Any, predictions: Any) -> None:
            super().process(targets, predictions)
            truth = [int(value) for value in torch.cat(targets).view(-1).tolist()]
            top1 = [int(value) for value in torch.cat(predictions)[:, 0].tolist()]
            self.macro_f1 = macro_f1_from_indices(truth, top1, self.class_count)

        @property
        def fitness(self) -> float:
            return self.macro_f1

        @property
        def keys(self) -> List[str]:
            return ["metrics/accuracy_top1", "metrics/accuracy_top5", "metrics/macro_f1"]

        @property
        def results_dict(self) -> Dict[str, float]:
            return dict(
                zip(self.keys + ["fitness"], [self.top1, self.top5, self.macro_f1, self.fitness])
            )

    class MacroF1Validator(ClassificationValidator):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.metrics = MacroF1Metrics()

        def init_metrics(self, model: Any) -> None:
            super().init_metrics(model)
            self.metrics.class_count = self.nc

        def get_desc(self) -> str:
            return ("%22s" + "%11s" * 3) % ("classes", "top1_acc", "top5_acc", "macro_f1")

        def print_results(self) -> None:
            ULTRALYTICS_LOGGER.info(
                ("%22s" + "%11.3g" * 3)
                % ("all", self.metrics.top1, self.metrics.top5, self.metrics.macro_f1)
            )

    class BalancedDistributedSampler(Sampler[int]):
        """Draw balanced classes independently on every DDP rank."""

        def __init__(self, weights: Any, replicas: int, rank: int, seed: int) -> None:
            self.weights = torch.as_tensor(weights, dtype=torch.double)
            self.replicas = replicas
            self.rank = rank
            self.seed = seed
            self.epoch = 0
            self.samples_per_rank = (len(self.weights) + replicas - 1) // replicas
            self.total_samples = self.samples_per_rank * replicas

        def __iter__(self) -> Iterable[int]:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            sampled = torch.multinomial(
                self.weights, self.total_samples, replacement=True, generator=generator
            ).tolist()
            return iter(sampled[self.rank : self.total_samples : self.replicas])

        def __len__(self) -> int:
            return self.samples_per_rank

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

    class RoleClassificationTrainer(ClassificationTrainer):
        validator_class = MacroF1Validator

        def build_optimizer(self, *args: Any, **kwargs: Any) -> Any:
            optimizer = super().build_optimizer(*args, **kwargs)
            return _disable_optimizer_foreach(optimizer) if backend == "xpu" else optimizer

        def resume_training(self, checkpoint: Any) -> None:
            super().resume_training(checkpoint)
            if backend == "xpu":
                _disable_optimizer_foreach(self.optimizer)

        def get_dataloader(
            self, dataset_path: Any, batch_size: int = 16, rank: int = 0, mode: str = "train"
        ) -> Any:
            with torch_distributed_zero_first(rank):
                dataset = self.build_dataset(dataset_path, mode)
            if mode != "train":
                loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)
            else:
                labels = [int(sample[1]) for sample in dataset.samples]
                weights = balanced_class_sample_weights(labels)
                counts: DefaultDict[int, int] = defaultdict(int)
                for label in labels:
                    counts[label] += 1
                ULTRALYTICS_LOGGER.info(
                    "Balanced role sampling: %s",
                    {str(index): counts[index] for index in sorted(counts)},
                )
                replicas = 1
                sampler_rank = 0
                generator = torch.Generator()
                generator.manual_seed(self.args.seed)
                if rank != -1 and torch.distributed.is_available() and torch.distributed.is_initialized():
                    replicas = torch.distributed.get_world_size()
                    sampler_rank = rank
                if replicas > 1:
                    sampler = BalancedDistributedSampler(weights, replicas, sampler_rank, self.args.seed)
                else:
                    sampler = WeightedRandomSampler(
                        weights, num_samples=len(labels), replacement=True, generator=generator
                    )
                worker_count = min(
                    (os.cpu_count() or 1) // max(torch.cuda.device_count(), 1), self.args.workers
                )
                loader = InfiniteDataLoader(
                    dataset=dataset,
                    batch_size=min(batch_size, len(dataset)),
                    shuffle=False,
                    num_workers=worker_count,
                    sampler=sampler,
                    pin_memory=PIN_MEMORY,
                    collate_fn=getattr(dataset, "collate_fn", None),
                    worker_init_fn=seed_worker,
                    generator=generator,
                )
            if mode != "train":
                if is_parallel(self.model):
                    self.model.module.transforms = loader.dataset.torch_transforms
                else:
                    self.model.transforms = loader.dataset.torch_transforms
            return loader

        def get_validator(self) -> Any:
            self.loss_names = ["loss"]
            return MacroF1Validator(
                self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
            )

    return RoleClassificationTrainer


def _configure_xpu_data_loading() -> None:
    """Disable page-locked host memory before and after Ultralytics imports."""
    os.environ["PIN_MEMORY"] = "false"
    try:
        from ultralytics.data import build as data_build
        from ultralytics.data import utils as data_utils
    except ImportError:
        return
    data_build.PIN_MEMORY = False
    data_utils.PIN_MEMORY = False


class TemporalRoleSmoother:
    """Exponential moving average of role probabilities for each track."""

    def __init__(self, alpha: float = 0.20) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha doit etre dans ]0, 1]")
        self.alpha = float(alpha)
        self._states: Dict[str, List[float]] = {}

    def update(self, track_key: str, probabilities: Sequence[float]) -> List[float]:
        values = [max(0.0, float(value)) for value in probabilities]
        total = sum(values)
        if not values or total <= 0:
            raise ValueError("Probabilites de role invalides")
        values = [value / total for value in values]
        previous = self._states.get(track_key)
        if previous is None:
            smoothed = values
        else:
            if len(previous) != len(values):
                raise ValueError("Nombre de classes incoherent pour {}".format(track_key))
            smoothed = [
                (1.0 - self.alpha) * old + self.alpha * new
                for old, new in zip(previous, values)
            ]
            normalizer = sum(smoothed)
            smoothed = [value / normalizer for value in smoothed]
        self._states[track_key] = smoothed
        return list(smoothed)

    def state(self, track_key: str) -> Optional[List[float]]:
        value = self._states.get(track_key)
        return list(value) if value is not None else None


def classification_metrics(
    truth: Sequence[str], predictions: Sequence[str], classes: Sequence[str]
) -> Dict[str, Any]:
    if len(truth) != len(predictions):
        raise ValueError("truth et predictions doivent avoir la meme longueur")
    labels = list(dict.fromkeys(classes))
    confusion = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual, predicted in zip(truth, predictions):
        if actual not in confusion or predicted not in confusion[actual]:
            continue
        confusion[actual][predicted] += 1
    per_class = {}
    f1_values = []
    correct = 0
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in labels if other != label)
        support = sum(confusion[label].values())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        correct += true_positive
        if support:
            f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    total = sum(sum(row.values()) for row in confusion.values())
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "samples": total,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def train_role_classifier(
    config_path: Path,
    hardware_override: Optional[Dict[str, Any]] = None,
    resume: bool = False,
) -> Any:
    """Fine-tune an Ultralytics classification model on prepared role crops."""
    config = resolve_relative_paths(load_yaml(config_path), config_path)
    hardware_config = dict(config.pop("hardware", {}))
    hardware_config.update(hardware_override or {})
    profile = resolve_profile(hardware_config)
    LOGGER.info(profile_summary(profile, inspect_gpus(backend=profile.backend)))
    model_name = config.pop("model", "yolo11s-cls.pt")
    resume_from = config.pop("resume_from", None)
    data = config.pop("data", None)
    if not data:
        raise ValueError("Le champ 'data' est obligatoire pour entrainer les roles")
    data_path = Path(str(data)).expanduser()
    if not data_path.is_absolute():
        data_path = data_path.resolve()
    if not data_path.is_dir():
        raise FileNotFoundError("Dataset de roles introuvable: {}".format(data_path))
    train_args = dict(config.pop("train", {}))
    if config:
        raise ValueError("Cles de configuration inconnues: {}".format(sorted(config)))
    if profile.backend == "xpu":
        _configure_xpu_data_loading()
        train_args["deterministic"] = bool(hardware_config.get("deterministic", False))
    train_args["device"] = profile.device
    train_args["amp"] = profile.amp
    train_args.setdefault("imgsz", profile.imgsz)
    if any(key in hardware_config for key in ("batch", "batch_per_device", "batch_per_gpu")):
        train_args["batch"] = profile.batch
    else:
        train_args.setdefault("batch", profile.batch)
    if "workers" in hardware_config:
        train_args["workers"] = profile.workers
    else:
        train_args.setdefault("workers", profile.workers)
    if resume:
        checkpoint = Path(str(resume_from or "")).expanduser() if resume_from else None
        if checkpoint is None:
            project = Path(str(train_args.get("project", "runs/classify"))).expanduser()
            checkpoint = project / str(train_args.get("name", "train")) / "weights" / "last.pt"
        if not checkpoint.is_absolute():
            checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Checkpoint de roles introuvable: {}".format(checkpoint))
        model_name = str(checkpoint)
    train_args.update(
        {
            "data": str(data_path),
            "task": "classify",
            "resume": resume,
        }
    )
    YOLO = _import_yolo(profile.backend)
    trainer = _role_classification_trainer(profile.backend)
    current = replace(
        profile,
        imgsz=int(train_args["imgsz"]),
        batch=int(train_args["batch"]),
        workers=int(train_args["workers"]),
    )
    for attempt in range(profile.max_oom_retries + 1):
        train_args.update(
            {
                "device": current.device,
                "imgsz": current.imgsz,
                "batch": current.batch,
                "workers": current.workers,
                "amp": current.amp,
            }
        )
        model = YOLO(str(model_name))
        try:
            LOGGER.info(
                "Entrainement roles tentative %s/%s: device=%s batch=%s amp=%s",
                attempt + 1,
                profile.max_oom_retries + 1,
                current.device,
                current.batch,
                current.amp,
            )
            return model.train(trainer=trainer, **train_args)
        except (RuntimeError, MemoryError, subprocess.CalledProcessError) as exc:
            retries_exhausted = attempt >= profile.max_oom_retries
            if not is_accelerator_oom(exc) or retries_exhausted:
                raise
            reduced = current.with_lower_memory_pressure()
            if reduced == current:
                raise
            LOGGER.warning(
                "Memoire %s insuffisante; nouvel essai des roles avec batch=%s",
                current.backend.upper(),
                reduced.batch,
            )
            current = reduced
        # Une nouvelle instance evite que le trainer/optimiseur en echec garde
        # de la VRAM XPU entre deux tentatives.
        del model
        gc.collect()
        try:
            empty_accelerator_cache(current.backend)
        except (ImportError, RuntimeError):
            pass
    raise RuntimeError("Entrainement des roles interrompu sans resultat")


def validate_role_classifier(
    weights: Path,
    data: Path,
    hardware: Optional[Dict[str, Any]] = None,
    split: str = "val",
    imgsz: int = 224,
    batch: int = 64,
) -> Any:
    if split not in {"val", "test"}:
        raise ValueError("Le split de classification doit etre val ou test")
    profile = resolve_profile(hardware)
    device = profile.primary_device
    YOLO = _import_yolo(profile.backend)
    model = YOLO(str(Path(weights).expanduser().resolve()))
    validation_args = {
        "data": str(Path(data).expanduser().resolve()),
        "split": split,
        "device": device,
        "imgsz": imgsz,
        "batch": batch,
    }
    if profile.half:
        validation_args["half"] = True
    trainer = _role_classification_trainer(profile.backend)
    return model.val(validator=trainer.validator_class, **validation_args)


def _model_names(model: Any) -> List[str]:
    names = model.names
    if isinstance(names, Mapping):
        ordered = sorted(names.items(), key=lambda item: int(item[0]))
        return [str(value) for _, value in ordered]
    return [str(name) for name in names]


def _argmax(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("Liste de scores vide")
    return max(range(len(values)), key=lambda index: values[index])


def _crop_path(dataset: Path, value: Any) -> Path:
    """Resolve a manifest crop path written on either Windows or POSIX."""
    relative = Path(str(value).replace("\\", "/"))
    return relative if relative.is_absolute() else dataset / relative


def _probability_mapping(names: Sequence[str], probabilities: Sequence[float]) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(names, probabilities)}


def _dataset_source_root(dataset: Path, override: Optional[Path]) -> Optional[Path]:
    if override is not None:
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError("Source SoccerNet introuvable: {}".format(candidate))
        return candidate
    metadata_path = dataset / "dataset.yaml"
    if not metadata_path.is_file():
        return None
    source = load_yaml(metadata_path).get("source")
    if not source:
        return None
    candidate = Path(str(source)).expanduser()
    if not candidate.is_absolute():
        candidate = (metadata_path.parent / candidate).resolve()
    return candidate if candidate.is_dir() else None


def _record_image_dimensions(
    record: Mapping[str, Any],
    source_root: Optional[Path],
    cache: Dict[str, Optional[Tuple[int, int]]],
) -> Optional[Tuple[int, int]]:
    try:
        width = int(record.get("image_width"))
        height = int(record.get("image_height"))
    except (TypeError, ValueError):
        width = 0
        height = 0
    if width > 0 and height > 0:
        return width, height
    source_image = str(record.get("source_image") or "").replace("\\", "/")
    if not source_image or source_root is None:
        return None
    if source_image in cache:
        return cache[source_image]
    path = source_root / Path(source_image)
    try:
        from PIL import Image

        with Image.open(str(path)) as image:
            dimensions = (int(image.width), int(image.height))
    except (FileNotFoundError, OSError):
        dimensions = None
    cache[source_image] = dimensions
    return dimensions


def _spatial_priors(
    manifest: Path,
    class_names: Sequence[str],
    source_root: Optional[Path],
    bins_x: int,
    bins_y: int,
    smoothing: float,
    dimension_cache: Dict[str, Optional[Tuple[int, int]]],
) -> Tuple[Any, Any, Dict[str, int]]:
    image_samples = []
    pitch_samples = []
    training_records = 0
    for record in read_jsonl(manifest):
        if record.get("split") != "train" or str(record.get("role")) not in class_names:
            continue
        training_records += 1
        role = str(record["role"])
        dimensions = _record_image_dimensions(record, source_root, dimension_cache)
        image_position = None
        if dimensions is not None:
            image_position = normalized_image_position(
                record.get("bbox_image") or {}, dimensions[0], dimensions[1]
            )
        pitch_position = normalized_pitch_position(record.get("bbox_pitch") or {})
        image_samples.append((role, spatial_prior_coordinates(image_position, "image")))
        pitch_samples.append((role, spatial_prior_coordinates(pitch_position, "pitch_oracle")))
    if not training_records:
        raise ValueError("Le manifeste ne contient aucun exemple train pour apprendre les priors")
    image_prior = fit_spatial_prior(
        image_samples,
        class_names,
        "image",
        bins_x=bins_x,
        bins_y=bins_y,
        smoothing=smoothing,
    )
    pitch_prior = fit_spatial_prior(
        pitch_samples,
        class_names,
        "pitch_oracle",
        bins_x=bins_x,
        bins_y=bins_y,
        smoothing=smoothing,
    )
    coverage = {
        "records": training_records,
        "image": sum(int(value) for value in image_prior.totals.values()),
        "pitch_oracle": sum(int(value) for value in pitch_prior.totals.values()),
    }
    return image_prior, pitch_prior, coverage


def predict_role_tracks(
    weights: Path,
    dataset: Path,
    output: Path,
    split: str = "val",
    manifest: Optional[Path] = None,
    hardware: Optional[Dict[str, Any]] = None,
    alpha: float = 0.20,
    imgsz: int = 224,
    batch: int = 64,
    max_samples: Optional[int] = None,
    source_root: Optional[Path] = None,
    image_prior_strength: float = 0.25,
    pitch_prior_strength: float = 0.75,
    spatial_bins_x: int = 12,
    spatial_bins_y: int = 8,
    spatial_smoothing: float = 1.0,
    quality_minimum_weight: float = 0.05,
) -> Dict[str, Any]:
    """Evaluate visual, quality-weighted and spatially-informed role tracks."""
    if split not in {"train", "val", "test"}:
        raise ValueError("Split de roles invalide: {}".format(split))
    if batch < 1:
        raise ValueError("batch doit etre >= 1")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples doit etre >= 1")
    if image_prior_strength < 0 or pitch_prior_strength < 0:
        raise ValueError("Les forces des priors doivent etre >= 0")
    if spatial_bins_x < 1 or spatial_bins_y < 1:
        raise ValueError("Les nombres de bins spatiaux doivent etre >= 1")
    if spatial_smoothing <= 0:
        raise ValueError("spatial_smoothing doit etre > 0")
    if not 0 < quality_minimum_weight <= 1:
        raise ValueError("quality_minimum_weight doit etre dans ]0, 1]")
    dataset = Path(dataset).expanduser().resolve()
    manifest = Path(manifest or dataset / "manifest.jsonl").expanduser().resolve()
    records = [record for record in read_jsonl(manifest) if record.get("split") == split]
    records.sort(
        key=lambda record: (
            str(record.get("sequence")),
            str(record.get("track_id")),
            int(record.get("frame_index", 0)),
        )
    )
    if max_samples is not None:
        records = records[:max_samples]
    if not records:
        raise ValueError("Aucun crop du split '{}' dans {}".format(split, manifest))
    for record in records:
        path = _crop_path(dataset, record["crop_path"])
        if not path.is_file():
            raise FileNotFoundError("Crop de role introuvable: {}".format(path))

    profile = resolve_profile(hardware)
    device = profile.primary_device
    YOLO = _import_yolo(profile.backend)
    model = YOLO(str(Path(weights).expanduser().resolve()))
    class_names = _model_names(model)
    unexpected = sorted(set(class_names) - set(ROLE_CLASSIFIER_ROLES))
    if unexpected:
        LOGGER.warning("Classes inattendues dans le modele de roles: %s", unexpected)
    resolved_source_root = _dataset_source_root(dataset, source_root)
    dimension_cache: Dict[str, Optional[Tuple[int, int]]] = {}
    prior_started = time.perf_counter()
    image_prior, pitch_prior, training_coverage = _spatial_priors(
        manifest,
        class_names,
        resolved_source_root,
        spatial_bins_x,
        spatial_bins_y,
        spatial_smoothing,
        dimension_cache,
    )
    prior_fit_seconds = time.perf_counter() - prior_started
    if training_coverage["image"] == 0:
        LOGGER.warning(
            "Prior image indisponible: regenerez le dataset ou utilisez --source-root pour "
            "retrouver les dimensions des images"
        )
    if split == "train":
        LOGGER.warning("Les priors sont appris et evalues sur train; ces metriques sont exploratoires")
    smoother = TemporalRoleSmoother(alpha)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "role_predictions.jsonl"
    raw_truth: List[str] = []
    raw_predictions: List[str] = []
    smooth_predictions: List[str] = []
    image_frame_predictions: List[str] = []
    pitch_frame_predictions: List[str] = []
    track_truth: DefaultDict[str, List[str]] = defaultdict(list)
    track_states: Dict[str, Dict[str, Any]] = {}
    track_metadata: Dict[str, Dict[str, Any]] = {}
    evaluation_coverage = {"image": 0, "pitch_oracle": 0}
    started = time.perf_counter()
    for start in range(0, len(records), batch):
        chunk = records[start : start + batch]
        sources = [str(_crop_path(dataset, record["crop_path"])) for record in chunk]
        predict_args = {
            "source": sources,
            "device": device,
            "imgsz": imgsz,
            "batch": batch,
            "verbose": False,
        }
        if profile.half:
            predict_args["half"] = True
        results = model.predict(**predict_args)
        if len(results) != len(chunk):
            raise RuntimeError("Ultralytics a retourne un nombre inattendu de predictions")
        for record, result in zip(chunk, results):
            if result.probs is None:
                raise RuntimeError("Le modele charge n'est pas un classifieur Ultralytics")
            probabilities = [float(value) for value in result.probs.data.cpu().tolist()]
            track_key = "{}:{}".format(record["sequence"], record["track_id"])
            smoothed = smoother.update(track_key, probabilities)
            dimensions = _record_image_dimensions(record, resolved_source_root, dimension_cache)
            image_position = None
            if dimensions is not None:
                image_position = normalized_image_position(
                    record.get("bbox_image") or {}, dimensions[0], dimensions[1]
                )
            pitch_position = normalized_pitch_position(record.get("bbox_pitch") or {})
            image_coordinates = spatial_prior_coordinates(image_position, "image")
            pitch_coordinates = spatial_prior_coordinates(pitch_position, "pitch_oracle")
            image_available = image_coordinates is not None and training_coverage["image"] > 0
            pitch_available = (
                pitch_coordinates is not None and training_coverage["pitch_oracle"] > 0
            )
            evaluation_coverage["image"] += int(image_available)
            evaluation_coverage["pitch_oracle"] += int(pitch_available)

            quality_image = getattr(result, "orig_img", None)
            if quality_image is None:
                from PIL import Image

                with Image.open(str(_crop_path(dataset, record["crop_path"]))) as crop:
                    quality = observation_quality(
                        crop.convert("RGB"),
                        probabilities,
                        image_position,
                        quality_minimum_weight,
                    )
            else:
                quality = observation_quality(
                    quality_image,
                    probabilities,
                    image_position,
                    quality_minimum_weight,
                )
            quality_weight = quality["weight"]
            image_probabilities = image_prior.probabilities(
                image_coordinates if image_available else None
            )
            pitch_probabilities = pitch_prior.probabilities(
                pitch_coordinates if pitch_available else None
            )
            fused_image = (
                fuse_with_spatial_prior(
                    smoothed, image_probabilities, image_prior_strength, quality_weight
                )
                if image_available
                else list(smoothed)
            )
            fused_pitch = (
                fuse_with_spatial_prior(
                    smoothed, pitch_probabilities, pitch_prior_strength, quality_weight
                )
                if pitch_available
                else list(smoothed)
            )

            raw_index = _argmax(probabilities)
            smooth_index = _argmax(smoothed)
            image_index = _argmax(fused_image)
            pitch_index = _argmax(fused_pitch)
            actual = str(record["role"])
            raw_role = class_names[raw_index]
            smooth_role = class_names[smooth_index]
            raw_truth.append(actual)
            raw_predictions.append(raw_role)
            smooth_predictions.append(smooth_role)
            image_frame_predictions.append(class_names[image_index])
            pitch_frame_predictions.append(class_names[pitch_index])
            track_truth[track_key].append(actual)
            track_metadata.setdefault(
                track_key,
                {
                    "sequence": str(record["sequence"]),
                    "track_id": str(record["track_id"]),
                },
            )
            state = track_states.setdefault(
                track_key,
                {
                    "visual": [],
                    "image": [],
                    "pitch_oracle": [],
                    "weights": [],
                    "payloads": [],
                    "image_coverage": 0,
                    "pitch_coverage": 0,
                },
            )
            state["visual"].append(smoothed)
            state["image"].append(fused_image)
            state["pitch_oracle"].append(fused_pitch)
            state["weights"].append(quality_weight)
            state["image_coverage"] += int(image_available)
            state["pitch_coverage"] += int(pitch_available)
            payload = dict(record)
            payload.update(
                {
                    "raw_role": raw_role,
                    "raw_confidence": probabilities[raw_index],
                    "smoothed_role": smooth_role,
                    "smoothed_confidence": smoothed[smooth_index],
                    "probabilities": _probability_mapping(class_names, probabilities),
                    "smoothed_probabilities": _probability_mapping(class_names, smoothed),
                    "quality": quality,
                    "image_position": dict(image_position or {}, available=image_available),
                    "pitch_position": dict(pitch_position or {}, available=pitch_available),
                    "image_prior_probabilities": (
                        _probability_mapping(class_names, image_probabilities)
                        if image_available
                        else None
                    ),
                    "pitch_oracle_prior_probabilities": (
                        _probability_mapping(class_names, pitch_probabilities)
                        if pitch_available
                        else None
                    ),
                    "image_fused_probabilities": _probability_mapping(class_names, fused_image),
                    "pitch_oracle_fused_probabilities": _probability_mapping(
                        class_names, fused_pitch
                    ),
                }
            )
            state["payloads"].append(payload)
    elapsed = time.perf_counter() - started
    track_actual: List[str] = []
    track_baseline_predicted: List[str] = []
    track_quality_predicted: List[str] = []
    track_image_predicted: List[str] = []
    track_pitch_predicted: List[str] = []
    track_prediction_path = output / "track_role_predictions.jsonl"
    with track_prediction_path.open("w", encoding="utf-8") as track_stream, prediction_path.open(
        "w", encoding="utf-8"
    ) as prediction_stream:
        for track_key, values in track_truth.items():
            counts: DefaultDict[str, int] = defaultdict(int)
            for value in values:
                counts[value] += 1
            actual = max(counts, key=counts.get)
            state = track_states[track_key]
            observations = len(state["visual"])
            baseline_probabilities = weighted_mean_probabilities(
                state["visual"], [1.0] * observations
            )
            quality_probabilities = weighted_mean_probabilities(
                state["visual"], state["weights"]
            )
            image_probabilities = weighted_mean_probabilities(
                state["image"], state["weights"]
            )
            pitch_probabilities = weighted_mean_probabilities(
                state["pitch_oracle"], state["weights"]
            )
            baseline_index = _argmax(baseline_probabilities)
            quality_index = _argmax(quality_probabilities)
            image_index = _argmax(image_probabilities)
            pitch_index = _argmax(pitch_probabilities)
            baseline_role = class_names[baseline_index]
            quality_role = class_names[quality_index]
            image_role = class_names[image_index]
            pitch_role = class_names[pitch_index]
            track_actual.append(actual)
            track_baseline_predicted.append(baseline_role)
            track_quality_predicted.append(quality_role)
            track_image_predicted.append(image_role)
            track_pitch_predicted.append(pitch_role)
            weight_sum = sum(state["weights"])
            weight_square_sum = sum(weight * weight for weight in state["weights"])
            effective_observations = (
                weight_sum * weight_sum / weight_square_sum if weight_square_sum else 0.0
            )

            def aggregation_payload(role: str, vector: Sequence[float]) -> Dict[str, Any]:
                index = _argmax(vector)
                return {
                    "role": role,
                    "confidence": vector[index],
                    "probabilities": _probability_mapping(class_names, vector),
                }

            payload = dict(track_metadata[track_key])
            payload.update(
                {
                    "track_key": track_key,
                    "observations": observations,
                    "role": actual,
                    # Backward-compatible fields remain the unweighted baseline.
                    "aggregated_role": baseline_role,
                    "aggregated_confidence": baseline_probabilities[baseline_index],
                    "mean_smoothed_probabilities": _probability_mapping(
                        class_names, baseline_probabilities
                    ),
                    "aggregations": {
                        "baseline": aggregation_payload(
                            baseline_role, baseline_probabilities
                        ),
                        "quality_weighted": aggregation_payload(
                            quality_role, quality_probabilities
                        ),
                        "image_prior": aggregation_payload(image_role, image_probabilities),
                        "pitch_oracle": aggregation_payload(pitch_role, pitch_probabilities),
                    },
                    "quality": {
                        "weight_sum": weight_sum,
                        "weight_mean": weight_sum / observations,
                        "weight_min": min(state["weights"]),
                        "weight_max": max(state["weights"]),
                        "effective_observations": effective_observations,
                    },
                    "spatial_coverage": {
                        "image": state["image_coverage"] / observations,
                        "pitch_oracle": state["pitch_coverage"] / observations,
                    },
                }
            )
            track_stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            for observation in state["payloads"]:
                observation["aggregation_contributions"] = {
                    "baseline": 1.0 / observations,
                    "quality_weighted": observation["quality"]["weight"] / weight_sum,
                    "image_prior": observation["quality"]["weight"] / weight_sum,
                    "pitch_oracle": observation["quality"]["weight"] / weight_sum,
                }
                prediction_stream.write(json.dumps(observation, ensure_ascii=False) + "\n")
    baseline_metrics = classification_metrics(
        track_actual, track_baseline_predicted, class_names
    )
    summary = {
        "split": split,
        "weights": str(Path(weights).expanduser().resolve()),
        "samples": len(records),
        "tracks": len(track_truth),
        "alpha": alpha,
        "track_aggregation": "mean_smoothed_probabilities",
        "elapsed_seconds": elapsed,
        "fps": len(records) / elapsed if elapsed else 0.0,
        "raw": classification_metrics(raw_truth, raw_predictions, class_names),
        "smoothed": classification_metrics(raw_truth, smooth_predictions, class_names),
        "image_prior_frame": classification_metrics(
            raw_truth, image_frame_predictions, class_names
        ),
        "pitch_oracle_frame": classification_metrics(
            raw_truth, pitch_frame_predictions, class_names
        ),
        "track_final": baseline_metrics,
        "track_baseline": baseline_metrics,
        "track_quality_weighted": classification_metrics(
            track_actual, track_quality_predicted, class_names
        ),
        "track_image_prior": classification_metrics(
            track_actual, track_image_predicted, class_names
        ),
        "track_pitch_oracle": classification_metrics(
            track_actual, track_pitch_predicted, class_names
        ),
        "postprocessing": {
            "prior_fit_split": "train",
            "prior_fit_seconds": prior_fit_seconds,
            "quality_formula": "fixed_v1",
            "quality_minimum_weight": quality_minimum_weight,
            "training_coverage": {
                name: value / training_coverage["records"]
                for name, value in training_coverage.items()
                if name != "records"
            },
            "evaluation_coverage": {
                name: value / len(records) for name, value in evaluation_coverage.items()
            },
            "image_prior": dict(
                image_prior.metadata(),
                strength=image_prior_strength,
                deployable_without_calibration=True,
            ),
            "pitch_oracle": dict(
                pitch_prior.metadata(),
                strength=pitch_prior_strength,
                deployable_without_calibration=False,
                warning=(
                    "Utilise bbox_pitch annote par SoccerNet; sert de borne haute et ne doit "
                    "pas etre utilise en production avant le module de calibration."
                ),
            ),
        },
        "predictions": str(prediction_path),
        "track_predictions": str(track_prediction_path),
    }
    (output / "role_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
