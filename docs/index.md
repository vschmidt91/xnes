# leitwerk

`leitwerk` is a small black-box optimization library built around a canonical xNES
update rule and a strict schema-first wrapper.

It is designed for expensive, stateful evaluation loops where you want to tune
scalar parameters, checkpoint progress, and optionally route mirrored samples
by context using `ask(context=...)`. Runtime parameters are exposed directly as
typed dataclass instances or nested plain dicts. Training keeps at most one
pending sample: call `ask()`, evaluate once, then call `tell()`. A second
`ask()` before `tell()` raises, and `tell()` without a pending `ask()` raises.
`save()` snapshots committed state only and does not persist a pending
reservation, so saving after `ask()` and reloading later rewinds to before that
`ask()`. `load()` may be called at any time; it replaces the current state and
cancels any pending sample. For deterministic inference, use `mean` to read
the current means without sampling.

The docs are split into two parts:

- a short usage path for people integrating the optimizer
- an API reference generated from the source code and docstrings

The public wrapper expects dataclass schemas whose optimized fields are declared
as `Annotated[float, Parameter(...)]`, with optional `min` and `max` bounds on
each scalar. State layout is lexicographic by field name, so declaration order
does not affect persistence, and saved schema definitions remain human-readable.
Context matching uses JSON-compatible labels normalized into stable strings for
mirror reuse and persistence. Loading a saved snapshot may intentionally
discard unsaved local progress, and in-flight evaluations are never resumable
from disk because pending reservations are ephemeral. For exact restart
semantics, checkpoint after `tell()`, not after `ask()`. Mean snapshots from
`mean` are returned directly as the schema type.
