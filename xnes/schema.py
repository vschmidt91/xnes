"""Schema parsing for the typed optimizer wrapper."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Annotated, Any, Generic, TypeVar, cast, get_args, get_origin, get_type_hints

import numpy as np
from scipy.special import expit, logit

T = TypeVar("T")
Path = tuple[str, ...]
BuildFn = Callable[[Mapping[Path, float]], Any]


class Parameter:
    """User-facing metadata for one optimized scalar schema field."""

    loc: float
    scale: float

    def __new__(cls, *args: Any, **kwargs: Any) -> Parameter:
        if cls is Parameter:
            return _UnboundedParameter(*args, **kwargs)
        return super().__new__(cls)

    @classmethod
    def unbounded(cls, *, loc: float = 0.0, scale: float = 1.0) -> Parameter:
        return _UnboundedParameter(loc=loc, scale=scale)

    @classmethod
    def between(
        cls,
        lower: float,
        upper: float,
        *,
        loc: float | None = None,
        scale: float = 1.0,
    ) -> Parameter:
        midpoint = 0.5 * (float(lower) + float(upper))
        return _BetweenParameter(
            lower=lower,
            upper=upper,
            loc=midpoint if loc is None else loc,
            scale=scale,
        )

    @classmethod
    def above(cls, lower: float, *, loc: float | None = None, scale: float = 1.0) -> Parameter:
        default_loc = float(lower) + float(_softplus(0.0))
        return _AboveParameter(
            lower=lower,
            loc=default_loc if loc is None else loc,
            scale=scale,
        )

    @classmethod
    def below(cls, upper: float, *, loc: float | None = None, scale: float = 1.0) -> Parameter:
        default_loc = float(upper) - float(_softplus(0.0))
        return _BelowParameter(
            upper=upper,
            loc=default_loc if loc is None else loc,
            scale=scale,
        )

    @classmethod
    def above_exponential(cls, lower: float, *, loc: float | None = None, scale: float = 1.0) -> Parameter:
        return _AboveExponentialParameter(
            lower=lower,
            loc=(float(lower) + 1.0) if loc is None else loc,
            scale=scale,
        )

    @classmethod
    def below_exponential(cls, upper: float, *, loc: float | None = None, scale: float = 1.0) -> Parameter:
        return _BelowExponentialParameter(
            upper=upper,
            loc=(float(upper) - 1.0) if loc is None else loc,
            scale=scale,
        )

    def decode_scalar(self, z: float) -> float:
        return float(self.forward(z))

    def forward(self, z: np.ndarray | float) -> np.ndarray:
        """Map latent coordinates into user-space values."""
        raise NotImplementedError

    def inverse_loc(self, x: float) -> float:
        """Map a user-space central value into latent coordinates."""
        raise NotImplementedError

    def validate(self, name: str) -> None:
        """Validate the parameter specification for one schema field."""
        raise NotImplementedError

    def state_payload(self) -> dict[str, object]:
        return {
            "kind": f"{type(self).__module__}.{type(self).__qualname__}",
            "spec": _json_normalize(asdict(cast(Any, self))),
        }

    def state_hash(self) -> str:
        blob = json.dumps(self.state_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.blake2b(blob, digest_size=16).hexdigest()

    def initial_state(self, name: str) -> tuple[float, float]:
        self.validate(name)
        mu0 = _coerce_finite(self.inverse_loc(self.loc), _field_component_name(name, "latent loc"))
        sigma0 = _coerce_positive(self.scale, _field_component_name(name, "scale"))
        return mu0, sigma0

    def _validate_loc_and_scale(self, name: str) -> None:
        _coerce_finite(self.loc, _field_component_name(name, "loc"))
        _coerce_positive(self.scale, _field_component_name(name, "scale"))


@dataclass(frozen=True, repr=False)
class _UnboundedParameter(Parameter):
    loc: float = 0.0
    scale: float = 1.0

    def forward(self, z: np.ndarray | float) -> np.ndarray:
        return np.asarray(z, dtype=float)

    def inverse_loc(self, x: float) -> float:
        return float(x)

    def validate(self, name: str) -> None:
        self._validate_loc_and_scale(name)

    def __repr__(self) -> str:
        return f"Parameter(loc={self.loc!r}, scale={self.scale!r})"


@dataclass(frozen=True, repr=False)
class _BetweenParameter(Parameter):
    lower: float
    upper: float
    loc: float
    scale: float = 1.0

    def forward(self, z: np.ndarray | float) -> np.ndarray:
        values = np.asarray(z, dtype=float)
        lower = float(self.lower)
        upper = float(self.upper)
        return lower + (upper - lower) * expit(values)

    def inverse_loc(self, x: float) -> float:
        lower = float(self.lower)
        upper = float(self.upper)
        return float(logit((float(x) - lower) / (upper - lower)))

    def validate(self, name: str) -> None:
        self._validate_loc_and_scale(name)
        lower = _coerce_finite(self.lower, _field_component_name(name, "lower"))
        upper = _coerce_finite(self.upper, _field_component_name(name, "upper"))
        loc = float(self.loc)
        if not lower < upper:
            msg = f"xnes schema field '{name}' must satisfy lower < upper for Parameter.between(...)"
            raise ValueError(msg)
        if not lower < loc < upper:
            msg = f"xnes schema field '{name}' must satisfy lower < loc < upper for Parameter.between(...)"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"Parameter.between(lower={self.lower!r}, upper={self.upper!r}, loc={self.loc!r}, scale={self.scale!r})"


@dataclass(frozen=True, repr=False)
class _AboveParameter(Parameter):
    lower: float
    loc: float
    scale: float = 1.0

    def forward(self, z: np.ndarray | float) -> np.ndarray:
        return float(self.lower) + _softplus(z)

    def inverse_loc(self, x: float) -> float:
        return float(_softplus_inverse(float(x) - float(self.lower)))

    def validate(self, name: str) -> None:
        self._validate_loc_and_scale(name)
        lower = _coerce_finite(self.lower, _field_component_name(name, "lower"))
        if not float(self.loc) > lower:
            msg = f"xnes schema field '{name}' must satisfy loc > lower for Parameter.above(...)"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"Parameter.above(lower={self.lower!r}, loc={self.loc!r}, scale={self.scale!r})"


@dataclass(frozen=True, repr=False)
class _BelowParameter(Parameter):
    upper: float
    loc: float
    scale: float = 1.0

    def forward(self, z: np.ndarray | float) -> np.ndarray:
        return float(self.upper) - _softplus(-np.asarray(z, dtype=float))

    def inverse_loc(self, x: float) -> float:
        return float(-_softplus_inverse(float(self.upper) - float(x)))

    def validate(self, name: str) -> None:
        self._validate_loc_and_scale(name)
        upper = _coerce_finite(self.upper, _field_component_name(name, "upper"))
        if not float(self.loc) < upper:
            msg = f"xnes schema field '{name}' must satisfy loc < upper for Parameter.below(...)"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"Parameter.below(upper={self.upper!r}, loc={self.loc!r}, scale={self.scale!r})"


@dataclass(frozen=True, repr=False)
class _AboveExponentialParameter(Parameter):
    lower: float
    loc: float
    scale: float = 1.0

    def forward(self, z: np.ndarray | float) -> np.ndarray:
        return float(self.lower) + np.exp(np.asarray(z, dtype=float))

    def inverse_loc(self, x: float) -> float:
        return float(np.log(float(x) - float(self.lower)))

    def validate(self, name: str) -> None:
        self._validate_loc_and_scale(name)
        lower = _coerce_finite(self.lower, _field_component_name(name, "lower"))
        if not float(self.loc) > lower:
            msg = f"xnes schema field '{name}' must satisfy loc > lower for Parameter.above_exponential(...)"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"Parameter.above_exponential(lower={self.lower!r}, loc={self.loc!r}, scale={self.scale!r})"


@dataclass(frozen=True, repr=False)
class _BelowExponentialParameter(Parameter):
    upper: float
    loc: float
    scale: float = 1.0

    def forward(self, z: np.ndarray | float) -> np.ndarray:
        return float(self.upper) - np.exp(-np.asarray(z, dtype=float))

    def inverse_loc(self, x: float) -> float:
        return float(-np.log(float(self.upper) - float(x)))

    def validate(self, name: str) -> None:
        self._validate_loc_and_scale(name)
        upper = _coerce_finite(self.upper, _field_component_name(name, "upper"))
        if not float(self.loc) < upper:
            msg = f"xnes schema field '{name}' must satisfy loc < upper for Parameter.below_exponential(...)"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"Parameter.below_exponential(upper={self.upper!r}, loc={self.loc!r}, scale={self.scale!r})"


@dataclass(frozen=True)
class SchemaDiff:
    """Difference between persisted and current schema identities.

    Attributes:
        added: Current schema leaf names absent from the loaded state.
        removed: Loaded schema leaf names absent from the current schema.
        changed: Shared leaf names whose persisted spec hash differs from the current schema.
        unchanged: Shared leaf names whose persisted spec hash matches the current schema.
    """

    added: list[str]
    removed: list[str]
    changed: list[str]
    unchanged: list[str]


@dataclass(frozen=True)
class FieldSpec:
    """Internal normalized description of one optimized schema leaf."""

    name: str
    path: Path
    parameter: Parameter
    mu0: float
    sigma0: float
    spec_hash: str

    def decode_scalar(self, z: float) -> float:
        return self.parameter.decode_scalar(z)


@dataclass(frozen=True)
class SchemaSpec(Generic[T]):
    """Internal parsed schema-tree description used by `Optimizer`."""

    model_type: type[T]
    fields: tuple[FieldSpec, ...]
    instantiate: Callable[[Mapping[Path, float]], T]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field_spec.name for field_spec in self.fields)

    @property
    def dim(self) -> int:
        return len(self.fields)

    def state_schema(self) -> dict[str, str]:
        return {field_spec.name: field_spec.spec_hash for field_spec in self.fields}

    def diff(self, saved_schema: Mapping[str, str]) -> SchemaDiff:
        current_schema = self.state_schema()
        current_names = self.names
        saved_names = sorted(saved_schema)
        added = [name for name in current_names if name not in saved_schema]
        removed = [name for name in saved_names if name not in current_schema]
        changed = [
            name for name in current_names if name in saved_schema and saved_schema[name] != current_schema[name]
        ]
        unchanged = [
            name for name in current_names if name in saved_schema and saved_schema[name] == current_schema[name]
        ]
        return SchemaDiff(added=added, removed=removed, changed=changed, unchanged=unchanged)

    def index_by_name(self) -> dict[str, int]:
        return {field_spec.name: idx for idx, field_spec in enumerate(self.fields)}

    def initial_distribution(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        loc = np.array([field_spec.mu0 for field_spec in self.fields], dtype=float)
        scale_diag = np.array([field_spec.sigma0 for field_spec in self.fields], dtype=float)
        step_size_path = np.zeros(self.dim, dtype=float)
        return loc, np.diag(scale_diag), step_size_path

    def build_params(self, values: np.ndarray) -> T:
        leaf_values = {
            field_spec.path: field_spec.decode_scalar(float(value))
            for field_spec, value in zip(self.fields, values, strict=True)
        }
        return self.instantiate(leaf_values)


def parse_schema(model_type: type[T]) -> SchemaSpec[T]:
    """Parse and validate a dataclass schema tree for schema-mode optimization.

    Internal nodes must be dataclasses and leaf fields must be declared as
    `Annotated[float, Parameter(...)]`. The resulting leaf specifications are
    ordered lexicographically by dotted path name so dataclass declaration
    order does not affect persisted state layout.
    """

    if not isinstance(model_type, type) or not is_dataclass(model_type):
        msg = "xnes schema must be a dataclass type."
        raise TypeError(msg)

    field_specs, instantiate = _parse_dataclass_type(model_type, ())
    return SchemaSpec(
        model_type=model_type,
        fields=tuple(sorted(field_specs, key=_field_name)),
        instantiate=cast(Callable[[Mapping[Path, float]], T], instantiate),
    )


def _parse_dataclass_type(model_type: type[Any], prefix: Path) -> tuple[tuple[FieldSpec, ...], BuildFn]:
    dataclass_fields = tuple(fields(model_type))
    type_hints = get_type_hints(model_type, include_extras=True)
    parsed_fields = tuple(
        _parse_dataclass_field(
            type_hints.get(field.name),
            prefix + (field.name,),
            field.init,
        )
        for field in dataclass_fields
    )
    field_specs = tuple(field_spec for child_specs, _ in parsed_fields for field_spec in child_specs)
    constructor = cast(Callable[..., Any], model_type)
    child_builders = tuple(
        (field.name, build) for field, (_, build) in zip(dataclass_fields, parsed_fields, strict=True)
    )

    def instantiate(values: Mapping[Path, float]) -> Any:
        kwargs = {name: build(values) for name, build in child_builders}
        return constructor(**kwargs)

    return field_specs, instantiate


def _parse_dataclass_field(annotation: Any, path: Path, init: bool) -> tuple[tuple[FieldSpec, ...], BuildFn]:
    name = _path_name(path)
    if not init:
        msg = f"xnes schema field '{name}' must be init=True"
        raise TypeError(msg)

    if get_origin(annotation) is Annotated:
        return _parse_leaf_field(annotation, path)

    if isinstance(annotation, type) and is_dataclass(annotation):
        return _parse_dataclass_type(annotation, path)

    msg = f"xnes schema field '{name}' must be annotated as Annotated[float, Parameter(...)] or be a dataclass type"
    raise TypeError(msg)


def _parse_leaf_field(annotation: Any, path: Path) -> tuple[tuple[FieldSpec, ...], BuildFn]:
    name = _path_name(path)
    runtime_type, *metadata = get_args(annotation)
    if runtime_type is not float:
        msg = f"xnes schema field '{name}' must be annotated as Annotated[float, Parameter(...)]"
        raise TypeError(msg)

    parameters = [item for item in metadata if isinstance(item, Parameter)]
    if len(parameters) != 1:
        msg = f"xnes schema field '{name}' must include exactly one Parameter(...) metadata value"
        raise TypeError(msg)

    field_spec = _normalize_field_spec(name, path, parameters[0])

    def instantiate(values: Mapping[Path, float]) -> Any:
        return float(values[path])

    return (field_spec,), instantiate


def _normalize_field_spec(name: str, path: Path, parameter: Parameter) -> FieldSpec:
    mu0, sigma0 = parameter.initial_state(name)
    return FieldSpec(
        name=name,
        path=path,
        parameter=parameter,
        mu0=mu0,
        sigma0=sigma0,
        spec_hash=parameter.state_hash(),
    )


def _field_name(field_spec: FieldSpec) -> str:
    return field_spec.name


def _path_name(path: Path) -> str:
    return ".".join(path)


def _field_component_name(field_name: str, component: str) -> str:
    return f"xnes schema field '{field_name}' {component}"


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


def _softplus(z: np.ndarray | float) -> np.ndarray:
    return np.logaddexp(0.0, np.asarray(z, dtype=float))


def _softplus_inverse(y: np.ndarray | float) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    return values + np.log1p(-np.exp(-values))
