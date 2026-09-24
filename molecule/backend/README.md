# ACE backend

The backend is a separate Python package and uses the existing ACE runtime in
the parent project's environment. Currently implemented: `GET
/api/v1/capabilities`, `/openapi.json`, and `/docs`. Job submission, evaluation,
and artifact endpoints remain subsequent implementation steps.

From the `molecule/` project root, after the scientific environment is set up:

```bash
uv sync --frozen --group backend --group test
uv run --frozen --group backend uvicorn ace_backend.app:create_app --factory --host 127.0.0.1 --port 8000
```

Query the API:

```bash
curl --fail-with-body http://localhost:8000/api/v1/capabilities
```

Capabilities describe runtime prerequisites, not whether a generation has
already succeeded. Each feature is checked independently during application
startup; the immutable result is reused for requests. Restart the server after
changing dependencies, model files, device visibility, or settings. The
capability checks do not deserialize checkpoints or run ACE sampling or docking.
When a separate inference worker is introduced, its readiness must become the
source of this response instead of the API process's startup snapshot.

Inference checks the selected CUDA device and required graph kernels, imports
the existing inference runtime, and checks its five required model/config files
for readable, nonempty content. Druglikeness checks the existing RDKit/SA scorer
and its fragment-score data. Scaffold evaluation checks RDKit. Docking checks
the existing QuickVina backend and runs QuickVina/Open Babel version commands
with a five-second timeout each. Missing prerequisites return a feature's
unavailability reason while the endpoint itself still returns `200`.

Server configuration uses these environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ACE_API_DEVICE` | `cuda:0` | Device index within the serving process's visible CUDA devices |
| `ACE_API_DISABLED_FEATURES` | `[]` | JSON array containing any of `inference`, `druglikeness`, `scaffold_preservation`, `docking` |
| `ACE_API_CORS_ORIGINS` | `[]` | JSON array of allowed browser origins, e.g. `["http://localhost:5173"]` |
| `ACE_API_LIMITS` | Contract defaults | JSON object with reduced operational limits, e.g. `{"max_num_samples": 8}` |

Disabled features are not probed. Invalid settings fail at startup. The preset
list remains the list of supported configurations even when inference is
unavailable. Responses include `X-Request-ID`; capabilities are not cached by
HTTP clients. The API contract is in
[`docs/WEB_DEMO_API_CONTRACT.md`](../docs/WEB_DEMO_API_CONTRACT.md).

Run the integration and configuration tests:

```bash
uv run --frozen --group backend --group test pytest backend/tests -q
uv run --frozen --group backend --group test ruff check backend
```

Tests exercise HTTP serialization, startup checks, CPU evaluation dependencies,
missing model files and tools, feature disabling, limits, CORS, request IDs, and
error responses. They hide physical GPUs and replace the GPU boundary for
successful and failing CUDA scenarios; model files in tests are temporary
fixtures. They do not validate full ACE inference.
