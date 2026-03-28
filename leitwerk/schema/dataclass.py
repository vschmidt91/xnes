"""Dataclass schema parsing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import Field, fields, is_dataclass
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

from .parameter import PARAMETER_METADATA_KEY, Parameter
from .parser import build_field_spec, build_scalar_builder, path_name
from .spec import BuildFn, FieldSpec, SchemaPath, SchemaSpec, T


def parse_dataclass_schema(model_type: type[T]) -> SchemaSpec[T]:
    """Parse and validate a dataclass schema tree."""
    if not isinstance(model_type, type) or not is_dataclass(model_type):
        msg = "leitwerk schema must be a dataclass type."
        raise TypeError(msg)

    field_specs, instantiate = _parse_dataclass_type(model_type, ())
    return SchemaSpec(
        fields=field_specs,
        instantiate=cast(Callable[[Mapping[SchemaPath, float]], T], instantiate),
    )


def _parse_dataclass_type(model_type: type[Any], prefix: SchemaPath) -> tuple[tuple[FieldSpec, ...], BuildFn]:
    dataclass_fields = tuple(fields(model_type))
    type_hints = get_type_hints(model_type, include_extras=True)
    parsed_fields = tuple(
        _parse_dataclass_field(
            type_hints.get(field.name),
            field,
            prefix + (field.name,),
        )
        for field in dataclass_fields
    )
    field_specs = tuple(field_spec for child_specs, _ in parsed_fields for field_spec in child_specs)
    constructor = cast(Callable[..., Any], model_type)
    child_builders = tuple(
        (field.name, build) for field, (_, build) in zip(dataclass_fields, parsed_fields, strict=True)
    )

    def instantiate(values: Mapping[SchemaPath, float]) -> Any:
        kwargs = {name: build(values) for name, build in child_builders}
        return constructor(**kwargs)

    return field_specs, instantiate


def _parse_dataclass_field(
    annotation: Any,
    dataclass_field: Field[Any],
    path: SchemaPath,
) -> tuple[tuple[FieldSpec, ...], BuildFn]:
    name = path_name(path)
    if not dataclass_field.init:
        msg = f"leitwerk schema field '{name}' must be init=True"
        raise TypeError(msg)

    if get_origin(annotation) is Annotated:
        return _parse_leaf_field(annotation, path)

    field_parameter = _field_parameter(dataclass_field)
    if field_parameter is not None:
        return _parse_parameter_field(annotation, field_parameter, path)

    if isinstance(annotation, type) and is_dataclass(annotation):
        return _parse_dataclass_type(annotation, path)

    msg = (
        f"leitwerk schema field '{name}' must be declared as float = parameter(...), "
        "annotated as Annotated[float, Parameter(...)], or be a dataclass type"
    )
    raise TypeError(msg)


def _field_parameter(parameter_field: Field[Any]) -> Parameter | None:
    value = parameter_field.metadata.get(PARAMETER_METADATA_KEY)
    if isinstance(value, Parameter):
        return value
    return None


def _parse_leaf_field(annotation: Any, path: SchemaPath) -> tuple[tuple[FieldSpec, ...], BuildFn]:
    name = path_name(path)
    runtime_type, *metadata = get_args(annotation)
    if runtime_type is not float:
        msg = f"leitwerk schema field '{name}' must be annotated as Annotated[float, Parameter(...)]"
        raise TypeError(msg)

    parameters = [item for item in metadata if isinstance(item, Parameter)]
    if len(parameters) != 1:
        msg = f"leitwerk schema field '{name}' must include exactly one Parameter(...) metadata value"
        raise TypeError(msg)

    field_spec = build_field_spec(path, parameters[0])
    return (field_spec,), build_scalar_builder(path)


def _parse_parameter_field(
    annotation: Any,
    field_parameter: Parameter,
    path: SchemaPath,
) -> tuple[tuple[FieldSpec, ...], BuildFn]:
    name = path_name(path)
    if annotation is not float:
        msg = f"leitwerk schema field '{name}' must be annotated as float when using parameter(...)"
        raise TypeError(msg)

    field_spec = build_field_spec(path, field_parameter)
    return (field_spec,), build_scalar_builder(path)
