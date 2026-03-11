"""Schema parsing for the typed optimizer wrapper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Annotated, Any, Generic, TypeVar, cast, get_args, get_origin, get_type_hints

import numpy as np

T = TypeVar("T")
Path = tuple[str, ...]
BuildFn = Callable[[Mapping[Path, float]], Any]


@dataclass(frozen=True)
class Prior:
    """Latent Gaussian prior for one optimized schema leaf.

    The optimizer operates in continuous latent space. For the currently
    supported schema mode, each leaf field maps directly to one latent float
    coordinate with mean `mean` and standard deviation `sigma`, while parent
    nodes may be nested dataclasses.
    """

    mean: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        mean = float(self.mean)
        sigma = float(self.sigma)
        if not np.isfinite(mean):
            msg = "Prior.mean must be a finite float."
            raise ValueError(msg)
        if not np.isfinite(sigma) or sigma <= 0.0:
            msg = "Prior.sigma must be a positive finite float."
            raise ValueError(msg)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "sigma", sigma)


@dataclass(frozen=True)
class FieldSpec:
    """Internal normalized description of one optimized schema leaf."""

    name: str
    path: Path
    runtime_type: type[float]
    prior: Prior


@dataclass(frozen=True)
class SchemaSpec(Generic[T]):
    """Internal parsed schema-tree description used by `Optimizer`."""

    model_type: type[T]
    fields: tuple[FieldSpec, ...]
    instantiate: Callable[[Mapping[Path, float]], T]


def parse_schema(model_type: type[T]) -> SchemaSpec[T]:
    """Parse and validate a dataclass schema tree for schema-mode optimization.

    Internal nodes must be dataclasses and leaf fields must be declared as
    `Annotated[float, Prior(...)]`. The resulting leaf specifications are
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

    msg = f"xnes schema field '{name}' must be annotated as Annotated[float, Prior(...)] or be a dataclass type"
    raise TypeError(msg)


def _parse_leaf_field(annotation: Any, path: Path) -> tuple[tuple[FieldSpec, ...], BuildFn]:
    name = _path_name(path)
    runtime_type, *metadata = get_args(annotation)
    if runtime_type is not float:
        msg = f"xnes schema field '{name}' must be annotated as Annotated[float, Prior(...)]"
        raise TypeError(msg)

    priors = [item for item in metadata if isinstance(item, Prior)]
    if len(priors) != 1:
        msg = f"xnes schema field '{name}' must include exactly one Prior(...) metadata value"
        raise TypeError(msg)

    field_spec = FieldSpec(name=name, path=path, runtime_type=float, prior=priors[0])

    def instantiate(values: Mapping[Path, float]) -> Any:
        return float(values[path])

    return (field_spec,), instantiate


def _field_name(field_spec: FieldSpec) -> str:
    return field_spec.name


def _path_name(path: Path) -> str:
    return ".".join(path)
