"""
preprocess.py — Prétraitement single-image pour le microservice PlumeID.

Pipeline : segmentation → débruitage → contraste (CLAHE) → padding 224×224

Entrée  : bytes bruts de l'image (tels que reçus par l'API)
Sortie  : PreprocessResult { ok, image, candidates, warning_code, user_message }
"""

import numpy as np
import cv2 as cv
from dataclasses import dataclass
from typing import Optional, List, Callable

# ── Constantes ────────────────────────────────────────────────
TARGET_SIZE = (224, 224)
MIN_MASK_RATIO = 0.005       # masque minimum = 0.5 % de l'image
CLOSE_KERNEL_SIZE = 5
MIN_COMPONENT_AREA = 500

# ── Codes de warning ─────────────────────────────────────────
WARNING_TOO_MANY_FEATHERS = "TOO_MANY_FEATHERS"
WARNING_NO_FEATHER = "NO_FEATHER"
WARNING_MULTIPLE_CANDIDATES = "MULTIPLE_CANDIDATES"


@dataclass
class PreprocessResult:
    """Résultat structuré du pipeline de prétraitement."""
    ok: bool
    image: Optional[np.ndarray]                  # 224×224 BGR si ok, None sinon
    candidates: Optional[List[np.ndarray]]       # liste si plusieurs masques
    warning_code: Optional[str]                  # code machine pour l'API
    user_message: Optional[str]                  # message à afficher côté app

    @property
    def status_code(self) -> int:
        return 200 if self.ok else 422


# ── 1. Segmentation ──────────────────────────────────────────
def _clean_mask(
    mask: np.ndarray,
    close_kernel: int = CLOSE_KERNEL_SIZE,
    min_area: int = MIN_COMPONENT_AREA,
) -> np.ndarray:
    """Fermeture morpho + suppression des petits composants."""
    if close_kernel > 0:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_CLOSE, k)

    n_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for lab in range(1, n_labels):
        if stats[lab, cv.CC_STAT_AREA] >= min_area:
            cleaned[labels == lab] = 1
    return cleaned


def _find_masks(image: np.ndarray) -> List[np.ndarray]:
    """
    Détecte tous les masques d'objets significatifs dans l'image.
    Retourne une liste de masques (uint8 0/1), triés par aire décroissante.
    """
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    binary = cv.adaptiveThreshold(
        blurred, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, 21, 5,
    )

    # Distance transform + watershed
    dist = cv.distanceTransform(binary, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    _, sure_fg = cv.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    sure_bg = cv.dilate(binary, kernel, iterations=3)
    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(cv.cvtColor(image, cv.COLOR_BGR2RGB), markers)

    min_area = h * w * MIN_MASK_RATIO
    masks = []

    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue
        mask = np.uint8(markers == marker_id)
        mask = _clean_mask(mask)
        area = int(mask.sum())
        if area >= min_area:
            masks.append((mask, area))

    # Fallback : contours si watershed n'a rien trouvé
    if not masks:
        contours, _ = cv.findContours(
            binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < min_area:
                continue
            m = np.zeros((h, w), dtype=np.uint8)
            cv.drawContours(m, [cnt], 0, 1, -1)
            masks.append((m, int(area)))

    # Tri par aire décroissante
    masks.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in masks]


def _crop_with_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Applique le masque (fond noir) et retourne le crop sur la bounding box."""
    mask_3ch = np.stack([mask] * 3, axis=-1)
    segmented = image * mask_3ch
    ys, xs = np.where(mask == 1)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return segmented[y1: y2 + 1, x1: x2 + 1]


# ── 2. Débruitage ────────────────────────────────────────────
def _denoise(image: np.ndarray) -> np.ndarray:
    return cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


# ── 3. Rehaussement de contraste (CLAHE) ─────────────────────
def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv.merge((l, a, b))
    return cv.cvtColor(enhanced, cv.COLOR_LAB2BGR)


# ── 4. Padding carré + resize ────────────────────────────────
def _pad_and_resize(image: np.ndarray, size: tuple = TARGET_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    max_side = max(h, w)

    # Fond noir carré
    canvas = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    x_off = (max_side - w) // 2
    y_off = (max_side - h) // 2
    canvas[y_off: y_off + h, x_off: x_off + w] = image

    return cv.resize(canvas, size, interpolation=cv.INTER_LANCZOS4)


# ── Messages utilisateur ──────────────────────────────────────
_MSG_TOO_MANY_FEATHERS = (
    "Plusieurs plumes ont été détectées sur l'image. "
    "Veuillez ne prendre qu'une seule plume en photo, "
    "ou reprendre votre photo sur un fond plus uni."
)

_MSG_NO_FEATHER = (
    "Aucune plume n'a été reconnue sur l'image. "
    "Veuillez reprendre la photo en suivant ces recommandations :\n"
    "• Utilisez un fond uni et contrasté (évitez les surfaces texturées)\n"
    "• Placez la plume à plat, seule, sans main visible\n"
    "• Cadrez pour que la plume occupe entre 30 % et 75 % de l'image"
)


# ── Traitement post-segmentation ─────────────────────────────
def _postprocess(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Applique crop + denoise + contraste + padding sur un masque."""
    cropped = _crop_with_mask(image, mask)
    denoised = _denoise(cropped)
    contrasted = _enhance_contrast(denoised)
    return _pad_and_resize(contrasted)


# ── Pipeline complet ─────────────────────────────────────────
def preprocess(image_bytes: bytes) -> PreprocessResult:
    """
    Point d'entrée unique.

    Retourne
    --------
    PreprocessResult avec 3 cas possibles :

    1 masque  → ok=True,  image=...,  candidates=None
    0 masque  → ok=False, image=None, warning_code=NO_FEATHER
    N masques → ok=False, image=None, candidates=[img1, …, imgN],
                warning_code=MULTIPLE_CANDIDATES
                → le microservice passe chaque candidat dans le modèle
                  et tranche (voir resolve_candidates).

    Raises
    ------
    ValueError
        Si les bytes ne peuvent pas être décodés en image.
    """
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv.imdecode(buf, cv.IMREAD_COLOR)
    if image is None:
        raise ValueError("Impossible de décoder l'image fournie.")

    masks = _find_masks(image)

    # ── 0 masque → rien trouvé ────────────────────────────
    if len(masks) == 0:
        return PreprocessResult(
            ok=False,
            image=None,
            candidates=None,
            warning_code=WARNING_NO_FEATHER,
            user_message=_MSG_NO_FEATHER,
        )

    # ── 1 masque → cas idéal ─────────────────────────────
    if len(masks) == 1:
        return PreprocessResult(
            ok=True,
            image=_postprocess(image, masks[0]),
            candidates=None,
            warning_code=None,
            user_message=None,
        )

    # ── N masques → préparer tous les candidats ──────────
    candidates = [_postprocess(image, m) for m in masks]
    return PreprocessResult(
        ok=False,
        image=None,
        candidates=candidates,
        warning_code=WARNING_MULTIPLE_CANDIDATES,
        user_message=None,  # pas de message user ici, le microservice décide
    )


def resolve_candidates(
    result: PreprocessResult,
    is_feather_fn: Callable[[np.ndarray], bool],
) -> PreprocessResult:
    """
    Appelée par le microservice quand result.warning_code == MULTIPLE_CANDIDATES.

    Paramètres
    ----------
    result : PreprocessResult
        Le résultat avec candidates rempli.
    is_feather_fn : callable(np.ndarray) -> bool
        Fonction qui prend une image 224×224 BGR et renvoie True si le modèle
        la classifie comme plume (pas "Non_plumes"). Fournie par le microservice.

    Retourne
    --------
    PreprocessResult
        - 1 plume trouvée    → ok=True, image=celle-là
        - 0 plume trouvée    → ok=False, NO_FEATHER
        - 2+ plumes trouvées → ok=False, TOO_MANY_FEATHERS
    """
    feathers = [img for img in result.candidates if is_feather_fn(img)]

    if len(feathers) == 1:
        return PreprocessResult(
            ok=True,
            image=feathers[0],
            candidates=None,
            warning_code=None,
            user_message=None,
        )

    if len(feathers) == 0:
        return PreprocessResult(
            ok=False,
            image=None,
            candidates=None,
            warning_code=WARNING_NO_FEATHER,
            user_message=_MSG_NO_FEATHER,
        )

    # 2+ plumes
    return PreprocessResult(
        ok=False,
        image=None,
        candidates=None,
        warning_code=WARNING_TOO_MANY_FEATHERS,
        user_message=_MSG_TOO_MANY_FEATHERS,
    )
