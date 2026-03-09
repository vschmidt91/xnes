# xnes

`xnes` is a small black-box optimization library built around a canonical xNES
update rule and a strict named-parameter wrapper.

It is designed for expensive, stateful evaluation loops where you want to tune
scalar parameters, checkpoint progress, and optionally route mirrored samples by
context using `ask(context=...)`.

The docs are split into two parts:

- a short usage path for people integrating the optimizer
- an API reference generated from the source code and docstrings

The implementation keeps parameter ordering lexicographic, so registration order
does not affect the state layout. Context matching uses explicit string labels,
while trial reservations remain runtime-only and are not persisted.
