# ACE backend

The backend is a separate Python package and uses the existing ACE runtime in
the parent project's environment. Implemented endpoints:

- `GET /api/v1/capabilities`
- `POST /api/v1/inference/jobs`
- `POST /api/v1/evaluation/jobs`
- `GET /api/v1/jobs/{job_id}` and `/result`
- `GET /api/v1/jobs/{job_id}/artifacts` and `/{artifact_id}`
- `/openapi.json` and `/docs`

Inference and evaluation share the persistent queue, status, result, and artifact endpoints.

From the `molecule/` project root, after the scientific environment is set up:

```bash
uv sync --frozen --group backend --group test
uv run --frozen --group backend uvicorn ace_backend.app:create_app --factory --workers 1 --host 127.0.0.1 --port 8000
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
Inference workers inherit the API process's Python interpreter, dependency
environment, CUDA device visibility, and configured device.

Inference checks the selected CUDA device and required graph kernels, imports
the existing inference runtime, and checks its five required model/config files
for readable, nonempty content. Druglikeness checks the existing RDKit/SA scorer
and its fragment-score data. Scaffold evaluation checks RDKit. Docking checks
the existing QuickVina backend and runs QuickVina/Open Babel version commands
with a five-second timeout each. Missing prerequisites return a feature's
unavailability reason while the endpoint itself still returns `200`.

Server configuration uses these environment variables:

| Variable                    | Default            | Meaning                                                                                         |
| --------------------------- | ------------------ | ----------------------------------------------------------------------------------------------- |
| `ACE_API_DEVICE`            | `cuda:0`           | Device index within the serving process's visible CUDA devices                                  |
| `ACE_API_DISABLED_FEATURES` | `[]`               | JSON array containing any of `inference`, `druglikeness`, `scaffold_preservation`, `docking`    |
| `ACE_API_CORS_ORIGINS`      | `[]`               | JSON array of allowed browser origins, e.g. `["http://localhost:5173"]`                         |
| `ACE_API_STORAGE_DIR`       | `outputs/web_demo` | Persistent job and artifact storage; relative paths resolve from the server's working directory |
| `ACE_API_LIMITS`            | Contract defaults  | JSON object with reduced operational limits, e.g. `{"max_num_samples": 8}`                      |

Disabled features are not probed. Invalid settings fail at startup. The preset
list remains the list of supported configurations even when inference is
unavailable. Responses include `X-Request-ID`; capabilities are not cached by
HTTP clients. The API contract is in
[`docs/WEB_DEMO_API_CONTRACT.md`](../docs/WEB_DEMO_API_CONTRACT.md).

## Inference submission and execution

Use the provided [`examples/inference-config.json`](../examples/inference-config.json)
and submit the three example files from the project root:

```bash
curl --fail-with-body http://localhost:8000/api/v1/inference/jobs \
  -F 'pocket_pdb=@examples/4m7t_pocket.pdb' \
  -F 'fragment_sdf=@examples/4m7t_fragment.sdf' \
  -F 'reference_ligand_sdf=@examples/4m7t_ligand.sdf' \
  -F 'config=<examples/inference-config.json'
```

`@path` uploads a file part. `<path` reads a file's contents into a regular form
field, so `config` contains JSON text rather than a file upload. Every scientific
config field must be supplied. Unknown and duplicate fields are rejected.

The server parses and sanitizes inputs, applies the default RDKit `RemoveHs`
policy, validates atom and byte limits, and checks the existing DiffSBDD pocket
selection before acceptance. Original uploads and prepared inputs are stored
separately under server-generated IDs. Prepared/generated SDFs use V3000 with
17 decimal places to preserve coordinates, including near the 8 Å pocket cutoff.

`202` returns the job links and `Location` header. Poll `links.self` every two
seconds until `succeeded` or `failed`; a succeeded job exposes `links.result`.
Available samples have SDF artifact URLs. File URLs support `?download=true`.
Input artifacts are already downloadable while the job is queued or running.

The Linux deployment uses one API serving process per storage directory, enforced
by an exclusive file lock. A background dispatcher executes one job at a time,
loading the existing four-expert ACE runtime in a fresh subprocess for each job.
The model process is independent of the submitting HTTP connection. Timeouts and
shutdown terminate the process group before another job can execute; a worker
also terminates if its parent API process dies. Do not use multiple Uvicorn workers
against the same storage directory.

The dispatcher persists job states and publishes complete files through a hashed
manifest. Completed jobs survive server restarts. Interrupted queued/running jobs
become `failed` with `service_restarted`, without automatic reruns. There is no
automatic expiration. Resolved configs, original/prepared inputs, code revisions
and local source changes, model hashes, execution environment, and results are
published as reproducibility artifacts. Internal worker logs remain private.

Sampling reuses `load_sampling_runtime` and `sample_condition`; it keeps the
existing coordinate restoration and molecule reconstruction. Each generated
molecule is sanitized and checked by SDF serialization/read-back before publication.
Invalid samples keep their original IDs. An all-invalid batch is still a completed
job with a `no_valid_samples` warning. Progress is `null` because the existing
sampler does not expose a progress callback.

## Evaluation submission and execution

Use [`examples/evaluation-config.json`](../examples/evaluation-config.json) to evaluate
an uploaded ligand with druglikeness, scaffold preservation, and QuickVina redocking:

```bash
curl --fail-with-body http://localhost:8000/api/v1/evaluation/jobs \
  -F 'ligand_sdf=@examples/4m7t_ligand.sdf' \
  -F 'fragment_sdf=@examples/4m7t_fragment.sdf' \
  -F 'pocket_pdb=@examples/4m7t_pocket.pdb' \
  -F 'reference_ligand_sdf=@examples/4m7t_ligand.sdf' \
  -F 'config=<examples/evaluation-config.json'
```

For druglikeness alone, use `metrics: ["druglikeness"]` and `docking: null`, and
upload only `ligand_sdf`. Druglikeness and topology accept 2D ligands; docking
requires finite 3D ligand/reference coordinates. CPU evaluations do not require
inference models or CUDA. Every requested metric must be available at submission.

To evaluate generated samples, set `source` to
`{"type":"inference_job","job_id":"<completed-inference-job-id>","sample_ids":[0]}`
and send only the `config` form field. Every selected sample must be `available`.
The evaluation job copies the source inputs, selected SDFs, settings, and provenance
before returning `202`, so later removal of the source job does not affect it.

Follow `links.self`, `links.result`, and `links.artifacts` as for inference.
Results have `kind: "evaluation"`, ascending sample IDs, and only the requested
metric groups. Each metric reports `succeeded` with a value, `failed` with an error,
or `skipped` for an unsanitizable ligand. A measured zero is a successful value.
Individual failures preserve other metrics; job timeouts and result-storage
failures fail the whole job. A succeeded evaluation may therefore contain failed
or skipped metrics.

Original files remain unchanged. Normalized input copies, result JSON, resolved
settings, and provenance are automatically stored and downloadable. Docking
records its explicit seed, actual box, Open Babel pH 7.4 preparation, tool versions,
and QuickVina executable hash. The original ligand coordinates are preserved;
the score is obtained by redocking, and v1 does not publish docked poses.

## Validation

Run the integration and configuration tests:

```bash
uv run --frozen --group backend --group test pytest backend/tests -q
uv run --frozen --group backend --group test ruff check backend
```

The default suite hides physical GPUs and substitutes model execution. It tests
all three example inputs through real multipart parsing, scientific preparation,
subprocess execution, status/results/downloads, hashes and coordinates, as well as
invalid inputs, queue saturation, storage errors, crashes, timeouts, and restart
recovery. A separate test compares the v1 preset against the existing inference
configuration. These tests do not establish actual model quality or CUDA behavior.

Evaluation tests run real RDKit metrics and worker subprocesses, verify source
snapshots after source-job removal and server restart, and cover strict inputs,
mixed inference/evaluation queue limits, individual metric failures, and worker
timeouts/crashes. The QuickVina smoke test runs real Open Babel/QuickVina when both
executables are installed and is skipped otherwise. Docking failure semantics are
also tested independently of external tools.

Run the opt-in real-model test separately with the intended CUDA device visible:

```bash
ACE_TEST_CUDA_DEVICE=cuda:0 uv run --frozen --group backend --group test pytest backend/tests/test_inference_cuda.py -q
```

This test uses real checkpoints and CUDA, one particle and 100 sampling steps,
with a five-minute execution timeout. It translates the example inputs together
to verify that generated coordinates return to the uploaded pocket's coordinate
frame, and checks result/artifact retrieval. It is an integration smoke test,
not an evaluation of generated ligand quality. It is skipped unless explicitly
enabled with `ACE_TEST_CUDA_DEVICE`.
