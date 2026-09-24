# Project Instructions

## Python coding guidelines

Before writing, modifying, or reviewing Python code, read
`docs/PYTHON_CODING_GUIDELINES.md` relative to the repository root
and follow its applicable rules.

Core principles:

- Prefer immutable configuration and value objects.
- Keep data separate from algorithms.
- Prefer Protocol and composition over deep inheritance.
- Use explicit types to constrain valid behavior and states.
- Validate inputs at boundaries and fail loudly on broken invariants.
- Avoid speculative abstractions and unnecessary framework wrappers.
- Make scientific assumptions explicit and preserve reproducibility.
- Test observable behavior and important integration boundaries.

Apply these rules to new and modified code.
Do not refactor unrelated code solely to enforce these guidelines.
