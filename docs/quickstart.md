# Quickstart

If you want to see `leitwerk` in a real loop, start with the SC2 bot example:

- [examples/train_sc2_bot.py](https://github.com/phantomsc2/leitwerk/blob/main/examples/train_sc2_bot.py)

Run it with:

```sh
python examples/train_sc2_bot.py
```

What the example does:

- tunes three scalar bot parameters with `OptimizerSession`
- samples one parameter set per game
- scores each game as `(outcome, efficiency)`
- uses enemy race as `ask(context=...)`
- writes optimizer state, history, and a rolling plot under `data/`

The high-level workflow is always the same:

1. define the search space with `Parameter(...)`
2. construct an `Optimizer` or `OptimizerSession`
3. call `ask(context=...)` to reserve one candidate
4. run exactly one evaluation with that candidate
5. call `tell(result)` to report the score
6. inspect `mean` or resume from saved state

Minimal loop:

```py
params = opt.ask(context)
result = evaluate(params)
report = opt.tell(result)
```

`ask()` / `tell()` is single-flight: one `ask()`, one evaluation, one `tell()`.

For the concrete API and persistence rules, continue with the
[Integration Guide](integration.md).
