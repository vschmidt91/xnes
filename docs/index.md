# leitwerk

`leitwerk` is a small black-box optimization library built around a canonical xNES
update rule and a strict schema-first wrapper.

It is designed for expensive, stateful evaluation loops where you want to tune
scalar parameters, checkpoint progress, and optionally route mirrored samples
by context using `ask(context=...)`. Runtime parameters are exposed directly as
typed dataclass instances or nested plain dicts. Training is strictly
sequential: call `ask()`, evaluate once, then call `tell()`. `save()` and
`load()` are only supported at idle boundaries, i.e. when no `ask()` is
pending. For deterministic inference, use `ask_best()` to read the current
means without sampling.

The docs are split into two parts:

- a short usage path for people integrating the optimizer
- an API reference generated from the source code and docstrings

The public wrapper expects dataclass schemas whose optimized fields are declared
as `Annotated[float, Parameter(...)]`, with optional `min` and `max` bounds on
each scalar. State layout is lexicographic by field name, so declaration order
does not affect persistence, and saved schema definitions remain human-readable.
Context matching uses JSON-compatible labels normalized into stable strings for
mirror reuse and persistence. Loading at an idle boundary may intentionally
discard unsaved local progress. Mean snapshots from `ask_best()` are returned
directly as the schema type.
