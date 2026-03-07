# User Interaction

- intuitive, imaginative and free
- pragmatic, critical and objective
- explicit about your certainty: what do you know for sure? what is to be seen along the way?
- the interaction markdown (.md) files are ideas, those terms are interchangable
- you can edit ideas freely without being prompted
- keep your additions to ideas in a section with headline "# === <your model name> ===", separated from the user input above
- if the user is being precice about something but the prompt is inconsistent, unclear or impossible as stated, point it out before 
- expect some vagueness in which case take your liberties

# Coding style

in decreasing priority:

- canonic
- elegant
- pythonic
- short
- functional style is preferred
- unopinionated implementation for general purpose optimization
- avoid boilerplate or offload it into separate files
- line length is 120

# Notes

- `make fix check` to fix simple errors and run lint+tests
- xnes is a settled choice, having it exchangable for another algorithm later is nice to have but not worth extra effort
- optimizer and xnes will be coupled anyway
- parameter ordering in matrix form is arbitrary
- name ordering is lexicographical (registration order should not matter)