# leitwerk

`leitwerk` is a schema-based evolutionary optimizer for long-running training loops.

## What It Does

- the knobs that can be turned, what they do, and how to measure success
- an optimizer that explores search spaces and learns from feedback
- a training loop when you wire the two together

## Who It Is For

- your code already runs in a loop or training setup that produces a score
- you want to save and resume progress and update your code in between
- each evaluation is expensive enough that sample efficiency matters

## Where Next

- start with the [README](https://github.com/phantomsc2/leitwerk) examples if you want to see it first
- read the [Integration Guide](guide.md) if you want to wire it into your own project
- use the [API Reference](reference/api.md) for exact signatures
- see the [FAQ](faq.md) for some tips and details
