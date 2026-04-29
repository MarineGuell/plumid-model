"""Plum'ID — inference package (HuggingFace model loader + predict)."""

from .classifier import (
    Classifier,
    ClassifierError,
    ClassifierStatus,
    get_classifier,
)

__all__ = [
    "Classifier",
    "ClassifierError",
    "ClassifierStatus",
    "get_classifier",
]
