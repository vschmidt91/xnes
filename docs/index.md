# xnes

`xnes` is a black-box optimizer with a dynamic named-parameter interface on top
of a canonical xNES update rule.

The docs are split into two parts:

- a short usage path for people integrating the optimizer
- an API reference generated from the source code and docstrings

The implementation keeps parameter ordering lexicographic, so registration order
does not affect the state layout.
