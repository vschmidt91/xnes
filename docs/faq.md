# FAQ

## What happens when the schema changes?

`leitwerk` keeps learned state for parameters that still match and resets only
what changed.

- parameters are identified by flattened names
- renaming a parameter resets that parameter
- changing `min` or `max` resets that parameter
- changing `mean` or `scale` changes the reset target, but does not force a reset
- adding or removing parameters does not reset the whole optimizer

If you use `OptimizerSession`, inspect `schema_diff` after loading to see what
changed.

## How should I choose the objective?

The objective usually matters more than the optimizer.

- put the real target first
- use later tuple items as deterministic tie-breakers
- keep objective semantics stable across long runs
- split genuinely different goals into separate optimizers

This is lexicographic ranking, not Pareto optimization.

## How does the optimizer work?

`leitwerk` uses a canonical xNES update underneath the schema wrapper. It
samples candidates from a multivariate normal, ranks them by result, and updates
the mean and scale with a natural-gradient step. Sampling is mirrored to reduce
noise, and `ask(context=...)` helps mirrored pairs stay aligned in stateful
environments.

For background on the update rule and the variance-reduction ideas, see:

- [Exponential Natural Evolution Strategies](https://people.idsia.ch/~tom/publications/xnes.pdf)
- [Mirrored Orthogonal Sampling with Pairwise Selection in Evolution Strategies](https://www.researchgate.net/publication/266087889_Mirrored_Orthogonal_Sampling_with_Pairwise_Selection_in_Evolution_Strategies)
