"""
inference/classifier.py
------------------------

Plum'ID image classifier.

* Downloads the model weights from a HuggingFace repo at first use.
* Auto-detects the network architecture by inspecting the state_dict's
  keys, so this code keeps working even if the team retrains with a
  different backbone (ResNet18/34/50/101, EfficientNet B0–B3, MobileNet
  V2 / V3, …) — without any code change here.
* Thread-safe singleton; lazy-loaded on the first prediction so the
  HTTP server can answer `/health` immediately and the cold-start cost
  is paid only when the first user image arrives.

Environment variables
---------------------
HF_REPO_ID              Required. HuggingFace repo, e.g. "Azerty112/Plum_ID_V1".
HF_MODEL_FILENAME       Optional. The .pth filename inside the repo.
                        If unset, the module picks the only .pth found, or
                        the most-recent one if there are several.
HF_REVISION             Optional. Git revision / branch / tag (default "main").
HF_TOKEN                Optional. Required when the repo is private.
MODEL_NUM_CLASSES       Optional. Forces the number of classes. When unset,
                        it's inferred from the last linear layer of the
                        state_dict — the safe default.
MODEL_CLASS_NAMES       Optional. Comma-separated species names, in the
                        same order as the model's output indices. Defaults
                        to the 6 seeded species. Length must match
                        MODEL_NUM_CLASSES.
MODEL_INPUT_SIZE        Optional integer (default 224). Edge length of the
                        square input the network expects.
MODEL_NORM_MEAN         Optional CSV (default ImageNet "0.485,0.456,0.406").
MODEL_NORM_STD          Optional CSV (default ImageNet "0.229,0.224,0.225").
HF_HOME                 Optional. Cache directory for HuggingFace files
                        (default /home/appuser/.cache/huggingface).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("plumid-model.classifier")


# --------------------------------------------------------------------- #
# Default species list (in the order assumed by the trained model).      #
# Override with MODEL_CLASS_NAMES if your model exposes a different      #
# ordering or set of classes.                                            #
# --------------------------------------------------------------------- #
DEFAULT_CLASS_NAMES: List[str] = [
    "Geai_des_chene_Passiform_garulus_glandarius",
    "Non_plumes",
    "Pic_epeiche_Dendrocopos_major",
    "Pie_bavarde_Pica_pica",
]


# --------------------------------------------------------------------- #
# Errors                                                                 #
# --------------------------------------------------------------------- #


class ClassifierError(RuntimeError):
    """Raised when the classifier cannot be loaded or run."""


# --------------------------------------------------------------------- #
# Status object (returned by /model/status)                              #
# --------------------------------------------------------------------- #


@dataclass
class ClassifierStatus:
    ready: bool
    architecture: Optional[str] = None
    num_classes: Optional[int] = None
    class_names: List[str] = field(default_factory=list)
    weights_file: Optional[str] = None
    repo_id: Optional[str] = None
    revision: Optional[str] = None
    input_size: Optional[int] = None
    device: Optional[str] = None
    last_load_seconds: Optional[float] = None
    last_error: Optional[str] = None


# --------------------------------------------------------------------- #
# Architecture detection                                                 #
# --------------------------------------------------------------------- #
# Heuristics keyed on layer-name patterns found in `state_dict.keys()`.
# We match in the order below and stop on the first hit. Each detector
# returns (arch_name, model_factory).
# --------------------------------------------------------------------- #


def _detect_architecture(state_keys: List[str], num_classes: int):
    """
    Look at the first / last keys of a state_dict to figure out which
    torchvision model was trained.

    Returns
    -------
    tuple : (architecture_name, factory_function)
        The factory takes `num_classes` and returns an *uninitialised*
        torch.nn.Module ready for `load_state_dict`.

    Raises
    ------
    ClassifierError if no architecture matches.
    """
    keys_set = set(state_keys)
    sample = "\n".join(state_keys[:30] + state_keys[-10:])
    log.debug("Detecting architecture from sample keys:\n%s", sample)

    # Lazy-import torchvision so this module can be loaded for tests on
    # machines that don't have torch installed.
    import torch.nn as nn
    import torchvision.models as tvm

    # ---------- ResNet family -----------------------------------------
    if (
        "conv1.weight" in keys_set
        and "bn1.weight" in keys_set
        and "fc.weight" in keys_set
        and any(k.startswith("layer4.") for k in state_keys)
    ):
        # ResNet variants are distinguished by the number of blocks
        # in layer1..layer4. Count unique "layerX.{idx}." prefixes.
        def _block_count(layer: str) -> int:
            idxs = set()
            for k in state_keys:
                if k.startswith(f"{layer}."):
                    idxs.add(k.split(".")[1])
            return len(idxs)

        layout = (_block_count("layer1"), _block_count("layer2"),
                  _block_count("layer3"), _block_count("layer4"))
        # Standard torchvision layouts:
        resnet_map = {
            (2, 2, 2, 2): ("resnet18", tvm.resnet18),
            (3, 4, 6, 3): ("resnet50", tvm.resnet50),  # could also be resnet34
            (3, 4, 23, 3): ("resnet101", tvm.resnet101),
            (3, 8, 36, 3): ("resnet152", tvm.resnet152),
        }
        # ResNet34 has the same layout as ResNet50 but with BasicBlock
        # (no `bn3` inside blocks). Disambiguate.
        has_bn3 = any(".bn3.weight" in k for k in state_keys)
        if layout == (3, 4, 6, 3) and not has_bn3:
            arch = "resnet34"
            factory = tvm.resnet34
        elif layout in resnet_map:
            arch, factory = resnet_map[layout]
        else:
            raise ClassifierError(
                f"ResNet-like state_dict but unknown layout: {layout}"
            )

        def _build(nc: int):
            m = factory(weights=None)
            in_features = m.fc.in_features
            m.fc = nn.Linear(in_features, nc)
            return m

        return arch, _build

    # ---------- EfficientNet (torchvision >=0.13) ---------------------
    if any(k.startswith("features.") for k in state_keys) and any(
        k.startswith("classifier.1.") for k in state_keys
    ):
        # EfficientNet variants share the same layer naming. We pick B0
        # by default; tweak via MODEL_EFFICIENTNET_VARIANT if needed.
        variant = os.environ.get("MODEL_EFFICIENTNET_VARIANT", "b0").lower()
        factory_map = {
            "b0": tvm.efficientnet_b0,
            "b1": tvm.efficientnet_b1,
            "b2": tvm.efficientnet_b2,
            "b3": tvm.efficientnet_b3,
        }
        if variant not in factory_map:
            raise ClassifierError(
                f"Unknown EfficientNet variant: {variant}"
            )

        def _build(nc: int):
            m = factory_map[variant](weights=None)
            in_features = m.classifier[1].in_features
            m.classifier[1] = nn.Linear(in_features, nc)
            return m

        return f"efficientnet_{variant}", _build

    # ---------- MobileNet V2 ------------------------------------------
    if any(k.startswith("features.0.0.weight") for k in state_keys) and any(
        k.startswith("classifier.1.") for k in state_keys
    ):
        # Distinguished from EfficientNet above by the absence of SE
        # blocks (no "block.X.X.fc" pattern).
        if not any(".block." in k for k in state_keys):
            def _build(nc: int):
                m = tvm.mobilenet_v2(weights=None)
                in_features = m.classifier[1].in_features
                m.classifier[1] = nn.Linear(in_features, nc)
                return m
            return "mobilenet_v2", _build

    # ---------- DenseNet (121 / 161 / 169 / 201) ----------------------
    # Signature: features.conv0.weight + features.norm5.weight (the final
    # BN before the classifier) + classifier.weight as a direct Linear.
    if (
        "features.conv0.weight" in keys_set
        and "features.norm5.weight" in keys_set
        and "classifier.weight" in keys_set
        and not any(k.startswith("classifier.1.") for k in state_keys)
    ):
        # Distinguish DenseNet variants by counting denselayers in
        # denseblock3 (the longest, most discriminative block):
        #   DenseNet121 → (6, 12, 24, 16)
        #   DenseNet161 → (6, 12, 36, 24)
        #   DenseNet169 → (6, 12, 32, 32)
        #   DenseNet201 → (6, 12, 48, 32)
        def _count_layers(block: str) -> int:
            prefix = f"features.{block}.denselayer"
            return len({
                k[len(prefix):].split(".")[0]
                for k in state_keys
                if k.startswith(prefix)
            })

        b3 = _count_layers("denseblock3")
        densenet_map = {
            24: ("densenet121", tvm.densenet121),
            36: ("densenet161", tvm.densenet161),
            32: ("densenet169", tvm.densenet169),  # also matches 201, see below
            48: ("densenet201", tvm.densenet201),
        }
        if b3 in densenet_map:
            arch, factory = densenet_map[b3]
            # Disambiguate 169 vs 201 — both have block3=32 vs 48
            # already handled by the dict.
        else:
            # Unknown count; default to densenet121 (most common).
            log.warning(
                "DenseNet detected but denseblock3 layer count is %d "
                "(expected 24/32/36/48). Defaulting to densenet121.", b3,
            )
            arch, factory = "densenet121", tvm.densenet121

        def _build(nc: int):
            m = factory(weights=None)
            in_features = m.classifier.in_features
            m.classifier = nn.Linear(in_features, nc)
            return m

        return arch, _build

    # ---------- MobileNet V3 (small / large) --------------------------
    # Signature: features.0.0.weight + classifier as a Sequential whose
    # final Linear is at index .3 (small) or .3 (large too).
    if any(k.startswith("features.0.0.weight") for k in state_keys) and any(
        k.startswith("classifier.3.") for k in state_keys
    ):
        # Distinguish small vs large by the number of inverted-residual
        # blocks in `features`. MobileNet V3 Small has 12 blocks total,
        # Large has 16.
        block_indices = {
            int(k.split(".")[1])
            for k in state_keys
            if k.startswith("features.") and k.split(".")[1].isdigit()
        }
        n_blocks = max(block_indices) + 1 if block_indices else 0
        if n_blocks >= 16:
            arch, factory = "mobilenet_v3_large", tvm.mobilenet_v3_large
        else:
            arch, factory = "mobilenet_v3_small", tvm.mobilenet_v3_small

        def _build(nc: int):
            m = factory(weights=None)
            in_features = m.classifier[-1].in_features
            m.classifier[-1] = nn.Linear(in_features, nc)
            return m

        return arch, _build

    # ---------- Fallback ----------------------------------------------
    # Dump the keys to help diagnose unknown architectures.
    log.error(
        "Cannot auto-detect architecture. State dict has %d keys. "
        "First 40 keys:\n%s\n…\nLast 10 keys:\n%s",
        len(state_keys),
        "\n".join(f"  {k}" for k in state_keys[:40]),
        "\n".join(f"  {k}" for k in state_keys[-10:]),
    )
    raise ClassifierError(
        "Cannot auto-detect the model architecture from the state_dict. "
        "See logs above for the dumped key list. "
        "Set MODEL_ARCHITECTURE to one of: "
        "resnet18, resnet34, resnet50, resnet101, resnet152, "
        "efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, "
        "mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, "
        "densenet121, densenet161, densenet169, densenet201 — "
        "or extend `_detect_architecture` in inference/classifier.py."
    )


def _build_explicit(arch: str, num_classes: int):
    """Build a model from an explicit architecture name (env override)."""
    import torch.nn as nn
    import torchvision.models as tvm

    arch = arch.lower().strip()
    if arch in {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}:
        factory = getattr(tvm, arch)
        m = factory(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch.startswith("efficientnet_"):
        factory = getattr(tvm, arch, None)
        if factory is None:
            raise ClassifierError(f"Unknown architecture: {arch}")
        m = factory(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if arch == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if arch in {"mobilenet_v3_small", "mobilenet_v3_large"}:
        factory = getattr(tvm, arch)
        m = factory(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if arch in {"densenet121", "densenet161", "densenet169", "densenet201"}:
        factory = getattr(tvm, arch)
        m = factory(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    raise ClassifierError(f"Unknown explicit architecture: {arch}")


# --------------------------------------------------------------------- #
# Number-of-classes inference                                            #
# --------------------------------------------------------------------- #


def _infer_num_classes(state: Dict[str, Any]) -> int:
    """
    Extract the number of output classes from the state_dict.

    The trained classifier head is always a linear layer in the models
    we support: `fc.weight` for ResNet, `classifier.1.weight` for
    EfficientNet / MobileNet. The first dimension of the weight tensor
    is the number of classes.
    """
    candidates = (
        "fc.weight",
        "classifier.1.weight",
        "classifier.3.weight",   # MobileNet V3
        "classifier.6.weight",   # VGG
        "classifier.weight",     # DenseNet (single Linear)
    )
    for key in candidates:
        if key in state:
            shape = tuple(state[key].shape)
            if len(shape) == 2:
                log.info("Inferred num_classes=%d from key %r", shape[0], key)
                return int(shape[0])
    raise ClassifierError(
        "Cannot infer the number of classes from the state_dict. "
        "Set MODEL_NUM_CLASSES explicitly."
    )


# --------------------------------------------------------------------- #
# HuggingFace download helper                                            #
# --------------------------------------------------------------------- #


def _download_weights(
    repo_id: str,
    filename: Optional[str],
    revision: str,
    token: Optional[str],
) -> Tuple[str, str]:
    """
    Resolve & download the .pth weights file from HuggingFace.

    Returns (local_path, resolved_filename).
    """
    from huggingface_hub import HfApi, hf_hub_download

    if filename:
        log.info("Downloading %s @ %s / %s", repo_id, revision, filename)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
        )
        return path, filename

    # No filename given — discover available .pth / .pt files.
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo_id, revision=revision)
    except Exception as exc:  # noqa: BLE001
        raise ClassifierError(
            f"Cannot list files in HuggingFace repo {repo_id!r}: {exc}"
        ) from exc

    weight_files = [f for f in files if f.lower().endswith((".pth", ".pt"))]
    if not weight_files:
        raise ClassifierError(
            f"No .pth or .pt file found in HuggingFace repo {repo_id!r}"
        )
    # Heuristic: pick the file with the highest accuracy in its name
    # (e.g. "model_V2_55%.pth"). Fallback to the first one.
    def _accuracy_key(name: str) -> int:
        import re
        m = re.search(r"(\d+)\s*%", name)
        return int(m.group(1)) if m else -1

    chosen = max(weight_files, key=_accuracy_key)
    log.info(
        "Auto-selected weights file %r (out of %d candidates)",
        chosen, len(weight_files),
    )
    path = hf_hub_download(
        repo_id=repo_id,
        filename=chosen,
        revision=revision,
        token=token,
    )
    return path, chosen


# --------------------------------------------------------------------- #
# Classifier                                                             #
# --------------------------------------------------------------------- #


class Classifier:
    """
    Lazy, thread-safe wrapper around a torch image classifier.

    Use `get_classifier()` to obtain the singleton instance.
    """

    def __init__(self) -> None:
        from .species_map import SpeciesMapper

        self._lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._device = None
        self._status = ClassifierStatus(ready=False)
        self._class_names: List[str] = []
        self._input_size: int = 224
        self._mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
        self._std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
        self._species_mapper = SpeciesMapper.from_env()

    # ----- public API ------------------------------------------------ #

    @property
    def status(self) -> ClassifierStatus:
        return self._status

    def ensure_loaded(self) -> None:
        """Idempotent; safe to call from multiple threads."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load()

    def predict(self, rgb_uint8: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        """
        Run inference on a preprocessed RGB image.

        Args
        ----
        rgb_uint8: ndarray of shape (H, W, 3), dtype uint8.
        top_k: how many candidates to return.

        Returns
        -------
        dict with keys:
            - model: "real"
            - architecture: e.g. "resnet50"
            - num_classes: total output classes
            - species_id: int — primary key in the API's species table
            - species_name: human-readable display name
            - model_class: raw label as emitted by the trained network
            - confidence: float in [0, 100] (percent), rounded to 2dp
            - top_k: list of {species_id, species_name, model_class, confidence}
        """
        self.ensure_loaded()
        if self._model is None:
            raise ClassifierError("Model failed to load; see /model/status")

        import torch
        import torch.nn.functional as F

        if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
            raise ClassifierError(
                f"Expected an HxWx3 RGB array, got shape {rgb_uint8.shape}"
            )

        # Resize if needed (defensive — preprocess_bytes already does
        # 224×224, but we don't want to crash if the caller skipped it).
        if (
            rgb_uint8.shape[0] != self._input_size
            or rgb_uint8.shape[1] != self._input_size
        ):
            import cv2 as cv
            rgb_uint8 = cv.resize(
                rgb_uint8,
                (self._input_size, self._input_size),
                interpolation=cv.INTER_LANCZOS4,
            )

        # uint8 [0,255] HWC RGB -> float32 [0,1] CHW, then normalize.
        arr = rgb_uint8.astype(np.float32) / 255.0
        arr = (arr - np.array(self._mean, dtype=np.float32)) / np.array(
            self._std, dtype=np.float32
        )
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        order = np.argsort(probs)[::-1]
        kk = min(top_k, len(self._class_names))

        # Build top-k entries with both the model's raw class label and
        # the resolved species record (id + display name).
        top: List[Dict[str, Any]] = []
        for idx in order[:kk]:
            i = int(idx)
            model_class = self._class_names[i]
            ref = self._species_mapper.resolve(model_class)
            top.append({
                "species_id": ref.id,
                "species_name": ref.display_name,
                "model_class": ref.model_class,
                "confidence": round(float(probs[i]) * 100.0, 2),
            })

        head = top[0]
        return {
            "model": "real",
            "architecture": self._status.architecture,
            "num_classes": len(self._class_names),
            "species_id": head["species_id"],
            "species_name": head["species_name"],
            "model_class": head["model_class"],
            "confidence": head["confidence"],
            "top_k": top,
        }

    def predict_class_name(self, rgb_uint8: np.ndarray) -> str:
        """
        Lightweight version of predict() that only returns the top-1
        model class name (e.g. "Geai_des_chene_Passiform_garulus_glandarius").

        Used by the preprocessing pipeline when several feather
        candidates are detected and we need to filter out the ones
        the model labels as "Non_plumes" before settling on the
        winning one.

        Returns
        -------
        str — the raw model class label of the top prediction.
        """
        import torch
        import torch.nn.functional as F

        self.ensure_loaded()
        if self._model is None:
            raise ClassifierError("Model failed to load; see /model/status")

        if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
            raise ClassifierError(
                f"Expected an HxWx3 array, got shape {rgb_uint8.shape}"
            )
        if (
            rgb_uint8.shape[0] != self._input_size
            or rgb_uint8.shape[1] != self._input_size
        ):
            import cv2 as cv
            rgb_uint8 = cv.resize(
                rgb_uint8,
                (self._input_size, self._input_size),
                interpolation=cv.INTER_LANCZOS4,
            )

        arr = rgb_uint8.astype(np.float32) / 255.0
        arr = (arr - np.array(self._mean, dtype=np.float32)) / np.array(
            self._std, dtype=np.float32
        )
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        top_idx = int(np.argmax(probs))
        return self._class_names[top_idx]

    def is_feather(self, image_bgr_or_rgb: np.ndarray) -> bool:
        """
        Return True if the model classifies the given image as a feather
        (i.e. not the 'Non_plumes' class). Convenient helper for
        `preprocess.resolve_candidates`.

        Note: assumes the input is RGB. If you have BGR (from OpenCV),
        convert first.
        """
        cls = self.predict_class_name(image_bgr_or_rgb)
        return not self._species_mapper.is_non_plume(cls)

    # ----- internal -------------------------------------------------- #

    def _load(self) -> None:
        t0 = time.perf_counter()
        try:
            import torch  # imported here so import failure is reported nicely

            # ------- Resolve config from the environment ------------- #
            repo_id = os.environ.get("HF_REPO_ID", "").strip()
            if not repo_id:
                raise ClassifierError(
                    "HF_REPO_ID is not set. Configure the HuggingFace repo "
                    "to download the model from."
                )
            filename = os.environ.get("HF_MODEL_FILENAME") or None
            revision = os.environ.get("HF_REVISION", "main").strip() or "main"
            token = os.environ.get("HF_TOKEN") or None

            self._input_size = int(os.environ.get("MODEL_INPUT_SIZE", "224"))
            self._mean = _parse_csv_floats(
                os.environ.get("MODEL_NORM_MEAN", "0.485,0.456,0.406")
            )
            self._std = _parse_csv_floats(
                os.environ.get("MODEL_NORM_STD", "0.229,0.224,0.225")
            )

            # ------- Download weights ------------------------------- #
            path, resolved_filename = _download_weights(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=token,
            )
            log.info("Weights cached at %s", path)

            # ------- Load state_dict -------------------------------- #
            try:
                # Prefer weights_only=True (torch >= 2.4) for safety.
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")

            # Some checkpoints wrap the state_dict in {"state_dict": ...}
            if isinstance(state, dict) and "state_dict" in state and all(
                isinstance(v, dict) for v in state.values()
            ):
                state = state["state_dict"]
            # Strip "module." prefixes left over from DataParallel.
            state = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in state.items()
            }

            # ------- Determine num_classes & architecture ----------- #
            forced_nc = os.environ.get("MODEL_NUM_CLASSES")
            num_classes = (
                int(forced_nc) if forced_nc else _infer_num_classes(state)
            )

            forced_arch = os.environ.get("MODEL_ARCHITECTURE", "").strip()
            if forced_arch:
                arch = forced_arch
                model = _build_explicit(arch, num_classes)
            else:
                arch, factory = _detect_architecture(
                    list(state.keys()), num_classes
                )
                model = factory(num_classes)

            # ------- Load weights into the freshly-built model ------ #
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                log.warning(
                    "load_state_dict had %d missing and %d unexpected keys "
                    "(first few: missing=%s unexpected=%s). Continuing.",
                    len(missing),
                    len(unexpected),
                    list(missing)[:3],
                    list(unexpected)[:3],
                )

            # ------- Class names ------------------------------------ #
            raw_names = os.environ.get("MODEL_CLASS_NAMES", "").strip()
            if raw_names:
                names = [s.strip() for s in raw_names.split(",") if s.strip()]
            else:
                names = list(DEFAULT_CLASS_NAMES)
            if len(names) != num_classes:
                log.warning(
                    "MODEL_CLASS_NAMES has %d entries but the model outputs "
                    "%d classes. Padding/truncating to match.",
                    len(names), num_classes,
                )
                if len(names) < num_classes:
                    names = names + [
                        f"class_{i}" for i in range(len(names), num_classes)
                    ]
                else:
                    names = names[:num_classes]
            self._class_names = names

            # ------- Device + eval mode ----------------------------- #
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            self._model = model
            self._device = device

            # ------- Status ----------------------------------------- #
            self._status = ClassifierStatus(
                ready=True,
                architecture=arch,
                num_classes=num_classes,
                class_names=names,
                weights_file=resolved_filename,
                repo_id=repo_id,
                revision=revision,
                input_size=self._input_size,
                device=device,
                last_load_seconds=round(time.perf_counter() - t0, 2),
                last_error=None,
            )
            self._loaded = True
            log.info(
                "Classifier ready: arch=%s num_classes=%d device=%s "
                "load_time=%.2fs",
                arch, num_classes, device, self._status.last_load_seconds,
            )

        except ClassifierError:
            self._status = ClassifierStatus(
                ready=False,
                last_error=str(self._status.last_error),
            )
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to load classifier")
            self._status = ClassifierStatus(
                ready=False,
                last_error=f"{type(exc).__name__}: {exc}",
            )
            raise ClassifierError(str(exc)) from exc


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def _parse_csv_floats(s: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise ClassifierError(
            f"Expected 3 comma-separated floats, got {s!r}"
        )
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


# --------------------------------------------------------------------- #
# Singleton accessor                                                     #
# --------------------------------------------------------------------- #


_singleton: Optional[Classifier] = None
_singleton_lock = threading.Lock()


def get_classifier() -> Classifier:
    """Return the process-wide Classifier instance (creates it if needed)."""
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = Classifier()
    return _singleton
