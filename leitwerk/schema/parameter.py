from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
from scipy.special import expit, logit

from .transforms import _softplus, _softplus_to_latent


@dataclass(frozen=True, slots=True)
class Parameter:
    """User-facing metadata for one optimized scalar schema field.

    `loc` is the user-space center value. `scale` is the latent-space standard
    deviation. `min` and `max`, when provided, are asymptotic user-space
    bounds applied through monotone coordinate transforms.
    """

    loc: float | None = None
    scale: float = 1.0
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "loc", None if self.loc is None else float(self.loc))
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "min", None if self.min is None else float(self.min))
        object.__setattr__(self, "max", None if self.max is None else float(self.max))

    def to_user_space(self, value: np.ndarray | float) -> np.ndarray:
        """Map latent coordinates into user-space values."""
        if self.min is None:
            if self.max is None:
                return np.asarray(value, dtype=float)
            return float(self.max) - _softplus(-np.asarray(value, dtype=float))
        if self.max is None:
            return float(self.min) + _softplus(value)
        values = np.asarray(value, dtype=float)
        return float(self.min) + (float(self.max) - float(self.min)) * expit(values)

    def loc_to_latent(self, value: float) -> float:
        """Map a user-space center value into latent coordinates."""
        if self.min is None:
            if self.max is None:
                return float(value)
            return float(-_softplus_to_latent(float(self.max) - float(value)))
        if self.max is None:
            return float(_softplus_to_latent(float(value) - float(self.min)))
        return float(logit((float(value) - float(self.min)) / (float(self.max) - float(self.min))))

    def validate(self, name: str) -> None:
        """Validate the parameter specification for one schema field."""
        self._validate_loc_and_scale(name)
        min_value = None if self.min is None else _coerce_finite(self.min, _field_component_name(name, "min"))
        max_value = None if self.max is None else _coerce_finite(self.max, _field_component_name(name, "max"))

        if min_value is not None and max_value is not None:
            if not min_value < max_value:
                msg = f"leitwerk schema field '{name}' must satisfy min < max for Parameter(...)"
                raise ValueError(msg)
            if self.loc is None:
                return
            if not min_value < float(self.loc) < max_value:
                msg = f"leitwerk schema field '{name}' must satisfy min < loc < max for Parameter(...)"
                raise ValueError(msg)
            return

        if self.loc is None:
            return

        if min_value is not None and not float(self.loc) > min_value:
            msg = f"leitwerk schema field '{name}' must satisfy loc > min for Parameter(...)"
            raise ValueError(msg)

        if max_value is not None and not float(self.loc) < max_value:
            msg = f"leitwerk schema field '{name}' must satisfy loc < max for Parameter(...)"
            raise ValueError(msg)

    def state_spec(self) -> dict[str, object]:
        return cast(dict[str, object], _json_normalize(asdict(self)))

    def reconciliation_key(self) -> tuple[float | None, float | None]:
        """Return the persisted transform data that must match to reuse latent state."""
        return self.min, self.max

    def initial_state(self, name: str) -> tuple[float, float]:
        self.validate(name)
        mean0 = (
            0.0
            if self.loc is None
            else _coerce_finite(
                self.loc_to_latent(float(self.loc)),
                _field_component_name(name, "latent mean"),
            )
        )
        scale0 = _coerce_positive(self.scale, _field_component_name(name, "scale"))
        return mean0, scale0

    def _validate_loc_and_scale(self, name: str) -> None:
        if self.loc is not None:
            _coerce_finite(self.loc, _field_component_name(name, "loc"))
        _coerce_positive(self.scale, _field_component_name(name, "scale"))


def _field_component_name(field_name: str, component: str) -> str:
    return f"leitwerk schema field '{field_name}' {component}"


def _coerce_finite(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        msg = f"{name} must be a finite float."
        raise ValueError(msg)
    return out


def _coerce_positive(value: float, name: str) -> float:
    out = _coerce_finite(value, name)
    if out <= 0.0:
        msg = f"{name} must be a positive finite float."
        raise ValueError(msg)
    return out


def _json_normalize(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    return value
