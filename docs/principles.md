# Core Principles

## 1. Generality

Black-box optimization is fundamentally domain-agnostic:
numbers go in, results come out, with no assumptions about what happens in between.
Evolutionary strategies are specifically designed for settings that are:

- random: the same input results in different outputs
- dynamic: the environment drifts during training
- ill-conditioned: parameters require very different scales and levels of precision

That being said, the documentation and examples reference StarCraft II in several places.
First, because that is where `leitwerk` was first developed.
Second, because it is a good example domain with all the properties above.

## 2. Sample Efficiency

Complex training loops spend almost all their time calculating the outcome.
In such settings, heavy-handed algorithms can be used without significantly affecting performance.
Instead, the main bottleneck is the number of training samples, so `leitwerk` aims to:

- provide state-of-the-art optimization strength
- preserve training progress whenever possible
- be quasi-parameter-free: restarts from bad settings are costly
