"""Mapping schema parsing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from .parameter import Parameter
from .parser import build_field_spec, build_scalar_builder, path_name
from .spec import BuildFn, FieldSpec, SchemaPath, SchemaSpec


def parse_mapping_schema(model: Mapping[str, object]) -> SchemaSpec[dict[str, Any]]:
    """Parse and validate a nested mapping schema."""
    field_specs, build_root = _parse_mapping_node(model, (), set())
    return SchemaSpec(
        fields=field_specs,
        instantiate=cast(Callable[[Mapping[SchemaPath, float]], dict[str, Any]], build_root),
    )


def _parse_mapping_node(
    model: Mapping[str, object],
    prefix: SchemaPath,
    seen_names: set[str],
) -> tuple[tuple[FieldSpec, ...], BuildFn]:
    parsed_fields = tuple(_parse_mapping_entry(key, value, prefix, seen_names) for key, value in model.items())
    field_specs = tuple(field_spec for _, child_specs, _ in parsed_fields for field_spec in child_specs)
    child_builders = tuple((name, build) for name, _, build in parsed_fields)

    def build_node(values: Mapping[SchemaPath, float]) -> dict[str, Any]:
        return {name: build(values) for name, build in child_builders}

    return field_specs, build_node


def _parse_mapping_entry(
    key: object,
    value: object,
    prefix: SchemaPath,
    seen_names: set[str],
) -> tuple[str, tuple[FieldSpec, ...], BuildFn]:
    if not isinstance(key, str):
        msg = "leitwerk schema mapping keys must be strings."
        raise TypeError(msg)

    path = prefix + (key,)
    _register_name(path, seen_names)
    if isinstance(value, Parameter):
        field_spec = build_field_spec(path, value)
        return key, (field_spec,), build_scalar_builder(path)

    if isinstance(value, Mapping):
        child_field_specs, build_node = _parse_mapping_node(cast(Mapping[str, object], value), path, seen_names)
        return key, child_field_specs, build_node

    name = path_name(path)
    msg = f"leitwerk schema field '{name}' must be Parameter(...) or a nested mapping"
    raise TypeError(msg)


def _register_name(path: SchemaPath, seen_names: set[str]) -> None:
    name = path_name(path)
    if name in seen_names:
        msg = f"leitwerk schema mapping path '{name}' is shadowed by another key."
        raise ValueError(msg)
    seen_names.add(name)
