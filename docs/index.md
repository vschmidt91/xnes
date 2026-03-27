# leitwerk

`leitwerk` is a schema-based evolutionary optimizer for long-running training loops.

## What does that mean?

- you define the knobs that can be turned, what they do and how to measure success
- the optimizer knows how to explore search spaces and learn from feedback
- wire this together, and you have a training loop

## Who is this for?

- you have code already running in a loop or training setup that produces a score
- you want to save and resume progress, and update your code in between
- you care about sample efficiency because evaluations are expensive

## Where to?

- start with the [README](https://github.com/phantomsc2/leitwerk) examples if you want to see it in action
- read the [Integration Guide](guide.md) if you want to use it into your own project
- use the [API Reference](reference/api.md) for exact signatures
- see the [FAQ](faq.md) for some tips and details
