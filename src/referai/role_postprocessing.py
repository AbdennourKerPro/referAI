"""Auditable quality weighting and spatial priors for person-role tracks."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def normalize_probabilities(probabilities: Sequence[float]) -> List[float]:
    values = [max(0.0, float(value)) for value in probabilities]
    total = sum(values)
    if not values or total <= 0:
        raise ValueError("Probabilites invalides")
    return [value / total for value in values]


def weighted_mean_probabilities(
    probabilities: Iterable[Sequence[float]], weights: Iterable[float]
) -> List[float]:
    """Aggregate observations while exposing their relative contribution."""
    vectors = [normalize_probabilities(values) for values in probabilities]
    values = [max(0.0, float(weight)) for weight in weights]
    if not vectors or len(vectors) != len(values):
        raise ValueError("Probabilites et poids doivent avoir la meme longueur non nulle")
    class_count = len(vectors[0])
    if any(len(vector) != class_count for vector in vectors):
        raise ValueError("Nombre de classes incoherent")
    total_weight = sum(values)
    if total_weight <= 0:
        raise ValueError("La somme des poids doit etre positive")
    totals = [0.0] * class_count
    for vector, weight in zip(vectors, values):
        totals = [total + weight * value for total, value in zip(totals, vector)]
    return normalize_probabilities([total / total_weight for total in totals])


def normalized_image_position(
    bbox_image: Mapping[str, Any], image_width: Any, image_height: Any
) -> Optional[Dict[str, float]]:
    """Describe a tracked person's ground contact in normalized image coordinates."""
    try:
        width = float(image_width)
        height = float(image_height)
        x_center = float(bbox_image["x_center"])
        y_center = float(bbox_image["y_center"])
        box_width = float(bbox_image["w"])
        box_height = float(bbox_image["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or box_width <= 0 or box_height <= 0:
        return None
    x_bottom = min(1.0, max(0.0, x_center / width))
    y_bottom = min(1.0, max(0.0, (y_center + box_height / 2.0) / height))
    normalized_width = min(1.0, max(0.0, box_width / width))
    normalized_height = min(1.0, max(0.0, box_height / height))
    return {
        "x_bottom": x_bottom,
        "y_bottom": y_bottom,
        "box_width": normalized_width,
        "box_height": normalized_height,
        "box_area": normalized_width * normalized_height,
        "aspect_ratio": box_width / box_height,
        "nearest_edge": min(x_bottom, 1.0 - x_bottom, y_bottom, 1.0 - y_bottom),
    }


def normalized_pitch_position(bbox_pitch: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """Normalize SoccerNet's metric pitch point while preserving useful distances."""
    try:
        x = float(bbox_pitch["x_bottom_middle"])
        y = float(bbox_pitch["y_bottom_middle"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    half_length = 52.5
    half_width = 34.0
    return {
        "x_m": x,
        "y_m": y,
        "x_normalized": min(1.0, max(0.0, (x + half_length) / (2.0 * half_length))),
        "y_normalized": min(1.0, max(0.0, (y + half_width) / (2.0 * half_width))),
        "goal_line_distance_m": max(0.0, half_length - abs(x)),
        "goal_line_distance_normalized": min(
            1.0, max(0.0, (half_length - abs(x)) / half_length)
        ),
        "lateral_distance_normalized": min(1.0, max(0.0, abs(y) / half_width)),
        "sideline_distance_m": max(0.0, half_width - abs(y)),
    }


def spatial_prior_coordinates(
    position: Optional[Mapping[str, float]], coordinate_system: str
) -> Optional[Tuple[float, float]]:
    if position is None:
        return None
    if coordinate_system == "image":
        return float(position["x_bottom"]), float(position["y_bottom"])
    if coordinate_system == "pitch_oracle":
        return (
            float(position["goal_line_distance_normalized"]),
            float(position["lateral_distance_normalized"]),
        )
    raise ValueError("Referentiel spatial inconnu: {}".format(coordinate_system))


def observation_quality(
    image: Any,
    probabilities: Sequence[float],
    image_position: Optional[Mapping[str, float]] = None,
    minimum_weight: float = 0.05,
) -> Dict[str, float]:
    """Compute fixed, interpretable visual and uncertainty quality indicators."""
    if not 0 < minimum_weight <= 1:
        raise ValueError("minimum_weight doit etre dans ]0, 1]")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy est requis pour calculer la qualite des crops") from exc

    array = np.asarray(image)
    if array.ndim == 3:
        grayscale = array.astype("float32").mean(axis=2)
    elif array.ndim == 2:
        grayscale = array.astype("float32")
    else:
        raise ValueError("Format d'image de crop invalide")
    height, width = grayscale.shape
    if height >= 3 and width >= 3:
        laplacian = (
            -4.0 * grayscale[1:-1, 1:-1]
            + grayscale[:-2, 1:-1]
            + grayscale[2:, 1:-1]
            + grayscale[1:-1, :-2]
            + grayscale[1:-1, 2:]
        )
        sharpness_raw = float(np.mean(laplacian * laplacian))
    else:
        sharpness_raw = 0.0
    contrast_raw = float(grayscale.std())
    brightness = float(grayscale.mean() / 255.0) if grayscale.size else 0.0
    sharpness = min(1.0, math.log1p(max(0.0, sharpness_raw)) / math.log1p(1000.0))
    contrast = min(1.0, max(0.0, contrast_raw / 64.0))
    exposure = min(1.0, max(0.0, 1.0 - abs(brightness - 0.5) / 0.5))
    resolution = min(1.0, math.sqrt(max(0, width * height)) / 128.0)
    box_area = float((image_position or {}).get("box_area", 0.0))
    geometry = min(1.0, math.sqrt(max(0.0, box_area)) / 0.12) if box_area else resolution

    normalized = normalize_probabilities(probabilities)
    class_count = len(normalized)
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in normalized)
    entropy = entropy / math.log(class_count) if class_count > 1 else 0.0
    ordered = sorted(normalized, reverse=True)
    confidence = ordered[0]
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    certainty = 1.0 - entropy

    # Appearance dominates so a confident but blurred error cannot dominate a track.
    score = (
        0.22 * sharpness
        + 0.13 * contrast
        + 0.10 * exposure
        + 0.18 * resolution
        + 0.12 * geometry
        + 0.15 * certainty
        + 0.07 * margin
        + 0.03 * confidence
    )
    weight = min(1.0, max(minimum_weight, score))
    return {
        "weight": weight,
        "sharpness": sharpness,
        "sharpness_raw": sharpness_raw,
        "contrast": contrast,
        "contrast_raw": contrast_raw,
        "brightness": brightness,
        "exposure": exposure,
        "resolution": resolution,
        "geometry": geometry,
        "entropy": entropy,
        "certainty": certainty,
        "top1_confidence": confidence,
        "top1_top2_margin": margin,
        "crop_width": float(width),
        "crop_height": float(height),
    }


@dataclass
class BinnedSpatialPrior:
    """Class-balanced, Laplace-smoothed spatial likelihood table."""

    roles: Tuple[str, ...]
    bins_x: int
    bins_y: int
    smoothing: float
    counts: Dict[str, List[float]]
    totals: Dict[str, float]
    coordinate_system: str

    def probabilities(self, coordinates: Optional[Tuple[float, float]]) -> List[float]:
        if coordinates is None:
            return [1.0 / len(self.roles)] * len(self.roles)
        index = self._bin_index(coordinates)
        number_of_bins = self.bins_x * self.bins_y
        likelihoods = [
            (self.counts[role][index] + self.smoothing)
            / (self.totals[role] + self.smoothing * number_of_bins)
            for role in self.roles
        ]
        return normalize_probabilities(likelihoods)

    def _bin_index(self, coordinates: Tuple[float, float]) -> int:
        x = min(1.0, max(0.0, float(coordinates[0])))
        y = min(1.0, max(0.0, float(coordinates[1])))
        x_bin = min(self.bins_x - 1, int(x * self.bins_x))
        y_bin = min(self.bins_y - 1, int(y * self.bins_y))
        return y_bin * self.bins_x + x_bin

    def metadata(self) -> Dict[str, Any]:
        return {
            "coordinate_system": self.coordinate_system,
            "bins": [self.bins_x, self.bins_y],
            "smoothing": self.smoothing,
            "training_samples": {role: int(self.totals[role]) for role in self.roles},
            "class_prior": "uniform",
        }


def fit_spatial_prior(
    samples: Iterable[Tuple[str, Optional[Tuple[float, float]]]],
    roles: Sequence[str],
    coordinate_system: str,
    bins_x: int = 12,
    bins_y: int = 8,
    smoothing: float = 1.0,
) -> BinnedSpatialPrior:
    if bins_x < 1 or bins_y < 1:
        raise ValueError("Le nombre de bins spatiaux doit etre >= 1")
    if smoothing <= 0:
        raise ValueError("Le lissage spatial doit etre > 0")
    ordered_roles = tuple(dict.fromkeys(str(role) for role in roles))
    if not ordered_roles:
        raise ValueError("Au moins un role est requis")
    number_of_bins = bins_x * bins_y
    prior = BinnedSpatialPrior(
        roles=ordered_roles,
        bins_x=bins_x,
        bins_y=bins_y,
        smoothing=float(smoothing),
        counts={role: [0.0] * number_of_bins for role in ordered_roles},
        totals={role: 0.0 for role in ordered_roles},
        coordinate_system=coordinate_system,
    )
    for role, coordinates in samples:
        if role not in prior.counts or coordinates is None:
            continue
        index = prior._bin_index(coordinates)
        prior.counts[role][index] += 1.0
        prior.totals[role] += 1.0
    return prior


def fuse_with_spatial_prior(
    visual_probabilities: Sequence[float],
    spatial_probabilities: Sequence[float],
    strength: float,
    quality: float,
) -> List[float]:
    """Fuse in log space, relying slightly more on space for weak observations."""
    if strength < 0:
        raise ValueError("La force du prior doit etre >= 0")
    visual = normalize_probabilities(visual_probabilities)
    spatial = normalize_probabilities(spatial_probabilities)
    if len(visual) != len(spatial):
        raise ValueError("Les distributions visuelle et spatiale sont incompatibles")
    quality = min(1.0, max(0.0, float(quality)))
    effective_strength = float(strength) * (1.25 - 0.75 * quality)
    logits = [
        math.log(max(value, 1e-12)) + effective_strength * math.log(max(prior, 1e-12))
        for value, prior in zip(visual, spatial)
    ]
    maximum = max(logits)
    return normalize_probabilities([math.exp(value - maximum) for value in logits])
