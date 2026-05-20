"""
inference/species_map.py
-------------------------

Maps model class names to canonical species records in the Plum'ID DB.

Why this matters
----------------
The network's output names follow a code-friendly convention
(`Geai_des_chene_Passiform_garulus_glandarius`), but the API and the
mobile app speak in terms of database IDs (`species_id`) and the
human-readable display names from the `species` table.

It supports three sources of mapping, in order of priority:

1. The `SPECIES_MAP` environment variable (JSON), e.g.
   {"Pic_epeiche_Dendrocopos_major": 2, "Non_plumes": 0}
2. A hardcoded default that mirrors the seed data in
   `db/initdb/01-schema.sql` (and Alembic revision 0002_seed_species).
3. A graceful fallback: any unknown class falls back to species_id 0
   ("Non identifié") rather than crashing.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

log = logging.getLogger("plumid-model.species_map")


# Default mapping — aligned with the rows seeded in the DB:
#   id 0 -> Non identifié
#   id 1 -> Pie bavarde
#   id 2 -> Pic épeiche
#   id 3 -> Perruche à collier
#   id 4 -> Geai des chênes
#   id 5 -> Corneille noire
#   id 6 -> Canard colvert
DEFAULT_NAME_TO_ID: Dict[str, int] = {
    "Non_plumes":                                   0,
    "Pie_bavarde_Pica_pica":                        1,
    "Pic_epeiche_Dendrocopos_major":                2,
    "Peruche_a_collier_Psittacula_krameri":         3,
    "Geai_des_chene_Passiform_garulus_glandarius":  4,
    "Corneille_noire_Corvus_corone":                5,
    "Canard_colvert_Anas_platyrhynchos":            6,
    # Spelling variants
    "Perruche_a_collier_Psittacula_krameri":        3,
    "Geai_des_chenes_Passiform_garulus_glandarius": 4,
}

DEFAULT_ID_TO_DISPLAY: Dict[int, str] = {
    0: "Non identifié",
    1: "Pie bavarde (Pica pica)",
    2: "Pic épeiche (Dendrocopos major)",
    3: "Perruche à collier (Psittacula krameri)",
    4: "Geai des chênes (Garrulus glandarius)",
    5: "Corneille noire (Corvus corone)",
    6: "Canard colvert (Anas platyrhynchos)",
}


@dataclass(frozen=True)
class SpeciesRef:
    """A canonical species reference, ready to send to the API/app."""
    id: int
    display_name: str
    model_class: str


class SpeciesMapper:
    """Resolves a model class name to a database species reference."""

    def __init__(
        self,
        name_to_id: Optional[Dict[str, int]] = None,
        id_to_display: Optional[Dict[int, str]] = None,
        unknown_id: int = 0,
    ) -> None:
        base_map = name_to_id or DEFAULT_NAME_TO_ID
        self._lookup_exact: Dict[str, int] = dict(base_map)
        self._lookup_lower: Dict[str, int] = {
            k.lower(): v for k, v in base_map.items()
        }
        self._id_to_display = id_to_display or DEFAULT_ID_TO_DISPLAY
        self._unknown_id = unknown_id

    @classmethod
    def from_env(cls) -> "SpeciesMapper":
        """
        Build a mapper from the environment.

        Reads SPECIES_MAP (optional JSON: {model_class_name: species_id})
        and SPECIES_DISPLAY_NAMES (optional JSON: {species_id: display}).
        Falls back to hardcoded defaults when unset.
        """
        raw_map = os.environ.get("SPECIES_MAP", "").strip()
        name_to_id: Optional[Dict[str, int]] = None
        if raw_map:
            try:
                parsed = json.loads(raw_map)
                name_to_id = {str(k): int(v) for k, v in parsed.items()}
                log.info(
                    "Using SPECIES_MAP from env (%d entries)",
                    len(name_to_id),
                )
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                log.warning(
                    "Cannot parse SPECIES_MAP env var (%s); falling back",
                    exc,
                )

        raw_disp = os.environ.get("SPECIES_DISPLAY_NAMES", "").strip()
        id_to_display: Optional[Dict[int, str]] = None
        if raw_disp:
            try:
                parsed = json.loads(raw_disp)
                id_to_display = {int(k): str(v) for k, v in parsed.items()}
                log.info(
                    "Using SPECIES_DISPLAY_NAMES from env (%d entries)",
                    len(id_to_display),
                )
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                log.warning(
                    "Cannot parse SPECIES_DISPLAY_NAMES env var (%s); falling back",
                    exc,
                )

        return cls(name_to_id=name_to_id, id_to_display=id_to_display)

    def resolve(self, model_class: str) -> SpeciesRef:
        """
        Translate a model class name to a SpeciesRef.

        Unknown classes are mapped to species_id=0 (Non identifié) with
        a warning. Matching is case-insensitive as a safety net.
        """
        if model_class is None:
            sid = self._unknown_id
        elif model_class in self._lookup_exact:
            sid = self._lookup_exact[model_class]
        elif model_class.lower() in self._lookup_lower:
            sid = self._lookup_lower[model_class.lower()]
        else:
            log.warning(
                "Unknown model class %r — falling back to species_id=%d",
                model_class, self._unknown_id,
            )
            sid = self._unknown_id

        return SpeciesRef(
            id=sid,
            display_name=self._id_to_display.get(sid, f"unknown(id={sid})"),
            model_class=model_class or "",
        )

    def is_non_plume(self, model_class: str) -> bool:
        """
        Return True if the given model_class maps to species_id=0
        (Non identifié / Non_plumes). Used by the preprocessing
        candidate-resolution logic.
        """
        return self.resolve(model_class).id == 0
