"""Plum'ID — inference package (HuggingFace model loader + predict)."""

from .classifier import (
    Classifier,
    ClassifierError,
    ClassifierStatus,
    get_classifier,
)
from .species_map import (
    DEFAULT_ID_TO_DISPLAY,
    DEFAULT_NAME_TO_ID,
    SpeciesMapper,
    SpeciesRef,
)

__all__ = [
    "Classifier",
    "ClassifierError",
    "ClassifierStatus",
    "DEFAULT_ID_TO_DISPLAY",
    "DEFAULT_NAME_TO_ID",
    "SpeciesMapper",
    "SpeciesRef",
    "get_classifier",
]
