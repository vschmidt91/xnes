# leitwerk

`leitwerk` is a schema-based evolutionary optimizer for long-running training loops.

## What?

- You define the knobs that can be turned, what they do and how to measure success.
- The optimizer knows how to explore search spaces and learn from feedback.
- Wire this together, and you have a training loop.

## Who is this for?

- your code already runs in a loop or training setup that produces a score
- you want to save and resume progress, and update your code in between
- each evaluation is expensive enough that sample efficiency matters

## Really not much more to it

- Start with the [README](https://github.com/phantomsc2/leitwerk) examples if you want to see it first
- Read the [Integration Guide](guide.md) if you want to wire it into your own project.
- Use the [API Reference](reference/api.md) for exact signatures.
- See the [FAQ](faq.md) for some tips and details.
