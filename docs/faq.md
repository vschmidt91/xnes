# FAQ

## What happens when the schema changes?

`leitwerk` keeps learned state for parameters that still match and resets only what changed:

- parameters are identified by flattened names
- renaming a parameter resets that parameter
- changing `min` or `max` resets that parameter
- changing `mean` or `scale` changes the reset target, but does not force a reset
- adding or removing parameters does not reset the others

`Optimizer.load()` and `OptimizerSession.schema_diff` report what changed as a `SchemaDiff`.

## What context should I provide?

If you are unsure what this does, provide none.

`leitwerk` creates samples in mirrored pairs, which is a stabilizing technique to keep the gradient estimate centered.
`ask(context=...)` lets both sides of the pair get evaluated in the same environment when possible.
Context values are matched by exact equality after JSON normalization.

For SC2 bots, useful contexts include:

- opponent race (consider delaying `ask` until scouting)
- own race (for random bots)
- map name
- opponent id

Batch size should be tuned together with the expected number of contexts:

- if batches are too small to encounter repeated contexts, matches will be rare
- rule of thumb: batch size should be at least twice the number of distinct contexts per batch

## How should I choose the objective?

For effective training, defining the objective matters more than the optimizer.

- put the primary objective first
- add tie-breakers for additional gradient information
- this is an encoding helper, not multi-objective / Pareto optimization
- only relative ranking matters, not absolute numeric values
- changing objectives mid-flight will leave the current batch with mixed signals
- split genuinely different goals into separate optimizers

## How does the optimizer work?

`leitwerk` provides a canonical xNES implementation.
Parameters are modeled as a multivariate normal distribution that is updated with natural-gradient steps.
The covariance matrix is estimated densely, initialization is diagonal.
Samples are generated with mirrored-orthogonal sampling for variance reduction.

Bounds are modeled as latent normals with smooth bijective activations:

- one-sided (`min` or `max`): affine-transformed softplus
- two-sided (`min` and `max`): affine-transformed sigmoid

Reference Papers:

- [Exponential Natural Evolution Strategies](https://people.idsia.ch/~tom/publications/xnes.pdf)
- [Mirrored Orthogonal Sampling with Pairwise Selection in Evolution Strategies](https://www.researchgate.net/publication/266087889_Mirrored_Orthogonal_Sampling_with_Pairwise_Selection_in_Evolution_Strategies)
