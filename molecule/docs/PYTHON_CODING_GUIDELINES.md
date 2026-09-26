# Python Coding Guidelines

## 0. Purpose

This guideline translates the programming philosophy of this project into concrete Python design rules.

The priorities are:

1. **Correctness**
2. **Readability**
3. **Type safety**
4. **Performance**
5. **Reproducibility**
6. **Explicitness**
7. **Development velocity**
8. **Extensibility**
9. **Simplicity**

Simplicity is not optimized directly. It should emerge from correct abstractions, readable code, and strong structural guarantees.

The overall philosophy is:

> Prefer statically understandable, behaviorally constrained, and easy-to-reason-about code.  
> Keep data and algorithms separate.  
> Use immutability and declarative style where they reduce mistakes, but allow controlled mutation and imperative internals when justified.  
> Abstract demonstrated structure, not hypothetical future needs.

---

# 1. State and Mutability

## 1.1 Prefer immutable public state

Objects representing configuration, domain values, and experiment definitions should usually be immutable after construction.

Prefer:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    temperature_kelvin: float
    timestep_fs: float
```

Avoid exposing arbitrary mutation:

```python
config.temperature_kelvin = 310  # avoid
```

### Rationale

Public mutable attributes allow callers to bypass validation and create states that the original author did not anticipate.

The goal is not to eliminate mutation entirely.

The goal is:

> **Do not expose uncontrolled mutation.**

---

## 1.2 Controlled mutation is acceptable

If mutation is necessary for performance, lifecycle management, or integration with an external framework, expose it through a narrow API.

Acceptable:

```python
class Simulation:
    def set_temperature(self, temperature_kelvin: float) -> None:
        if temperature_kelvin <= 0:
            raise ValueError("temperature must be positive")
        self._temperature_kelvin = temperature_kelvin
```

Strongly avoid:

```python
simulation.temperature_kelvin = -10
```

Prefer controlled setters only when state mutation is genuinely part of the object's lifecycle.

If many setters accumulate, prefer configuration-as-data instead.

---

## 1.3 Local mutation is acceptable

Local mutation inside a function is not inherently problematic.

This is acceptable:

```python
def transform_all(items: list[Input]) -> list[Output]:
    results: list[Output] = []

    for item in items:
        results.append(transform(item))

    return results
```

Do not contort code into a purely immutable style if that reduces clarity or performance.

The relevant boundary is externally observable state, not whether a local list uses `append`.

---

# 2. Declarative vs Imperative Code

## 2.1 Prefer declarative transformations when they improve reasoning

Prefer expressing _what_ transformation is performed rather than manually managing state when the declarative form is easier to understand.

Example:

```python
energies = [
    calculate_energy(structure)
    for structure in structures
    if is_valid(structure)
]
```

This is generally preferable to:

```python
energies = []

for structure in structures:
    if is_valid(structure):
        energy = calculate_energy(structure)
        energies.append(energy)
```

when both are equally readable.

---

## 2.2 Do not optimize for functional purity

Avoid forcing `map`, `filter`, `reduce`, or deeply chained functional constructs when they make code harder to read.

Avoid clever code such as:

```python
result = reduce(
    combine,
    map(transform, filter(predicate, values)),
)
```

when an explicit loop or intermediate variables make the control flow clearer.

---

## 2.3 Intermediate variables should carry meaning

Introduce an intermediate variable when:

- it is reused;
- it represents a meaningful concept;
- naming it improves local reasoning;
- it helps debugging or inspection.

Do not mechanically name every one-shot transformation.

Prefer:

```python
valid_structures = [x for x in structures if is_valid(x)]
scores = score_structures(valid_structures)
```

when `valid_structures` is conceptually meaningful.

A direct expression is fine when the intermediate state adds no useful meaning.

---

# 3. Data and Algorithms

## 3.1 Keep data/state separate from algorithms

Domain objects should primarily represent stable data, intrinsic properties, and intrinsic capabilities.

Algorithms that operate on those objects should usually be free functions, modules, or services.

Prefer:

```python
ligand = Ligand(...)
pose = dock(ligand, protein, docking_config)
```

over:

```python
pose = ligand.dock(protein, docking_config)
```

when docking is an external algorithm rather than an intrinsic property of a ligand.

---

## 3.2 Use methods for intrinsic capabilities

A method is appropriate when the operation belongs fundamentally to the type.

Examples that naturally belong on `Molecule`:

```python
molecule.num_atoms()
molecule.formal_charge()
molecule.canonicalize()
```

Examples that usually do not:

```python
generate_conformers(molecule)
dock(molecule, protein)
calculate_partial_charge(molecule, method)
run_md(molecule)
```

Rule of thumb:

> **Objects own intrinsic knowledge. Algorithms operate on objects.**

---

# 4. Dataclasses and Domain Types

## 4.1 Prefer `dataclass` for structured data

Use `dataclass` for explicit structured state.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Ligand:
    smiles: str
    formal_charge: int
```

Use `frozen=True` by default for configuration and value-like domain objects.

---

## 4.2 Do not create wrapper types for every semantic distinction

Avoid excessive nominal wrappers if they do not meaningfully constrain behavior.

Usually unnecessary:

```python
class Kelvin(float):
    ...

class Angstrom(float):
    ...

class KcalPerMol(float):
    ...
```

Prefer clear names:

```python
temperature_kelvin: float
distance_angstrom: float
binding_energy_kcal_mol: float
```

unless unit-aware types solve a concrete recurring correctness problem.

---

## 4.3 Use types aggressively for behavioral/state distinctions

Different states should become different types when they carry different valid data or support different operations.

Prefer:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Prepared:
    system: System


@dataclass(frozen=True)
class Running:
    job_id: str


@dataclass(frozen=True)
class Completed:
    result: SimulationResult


@dataclass(frozen=True)
class Failed:
    error: Exception


type SimulationState = Prepared | Running | Completed | Failed
```

over:

```python
@dataclass
class Simulation:
    status: str
    result: SimulationResult | None
    error: Exception | None
```

when combinations such as:

```python
status == "completed"
result is None
```

would otherwise be representable.

Rule:

> **Use types to encode real differences in behavior and legal state, not semantic trivia.**

---

# 5. Protocols and Polymorphism

## 5.1 Prefer `Protocol` to implementation inheritance

Prefer capability-oriented interfaces.

```python
from typing import Protocol


class SimulationBackend(Protocol):
    def run(
        self,
        system: System,
        config: SimulationConfig,
    ) -> SimulationResult:
        ...
```

Implementations do not need a shared base class:

```python
class OpenMMBackend:
    def run(
        self,
        system: System,
        config: SimulationConfig,
    ) -> SimulationResult:
        ...
```

---

## 5.2 Avoid deep inheritance hierarchies

Avoid:

```python
BaseSimulation
    -> MolecularSimulation
        -> DynamicsSimulation
            -> MDSimulation
                -> OpenMMMDSimulation
```

Prefer:

- `Protocol`;
- composition;
- explicit data types;
- sum/union types.

---

## 5.3 Cheap structural abstractions are acceptable early

YAGNI does not mean every abstraction must wait for a second implementation.

A small `Protocol` is acceptable even with one implementation when it:

- makes a capability relationship explicit;
- improves static checking;
- documents a boundary;
- has negligible maintenance cost.

Do not, however, build elaborate adapter/factory/provider layers for hypothetical future implementations.

---

# 6. Static Typing

## 6.1 Treat static typing as a design tool

Type annotations are not optional documentation. They are part of the program structure.

Prefer APIs whose result type is obvious from the signature.

Strongly prefer:

```python
def load_protein(path: Path) -> Protein:
    ...


def load_ligand(path: Path) -> Ligand:
    ...


def load_trajectory(path: Path) -> Trajectory:
    ...
```

over:

```python
def load(path: Path) -> Protein | Ligand | Trajectory:
    ...
```

when the caller already knows what kind of object is expected.

---

## 6.2 Avoid `Any` unless crossing an unavoidable dynamic boundary

Prefer specific types.

Avoid:

```python
def parse(config: dict[str, Any]) -> Any:
    ...
```

when a typed configuration object can be defined.

---

## 6.3 Prefer exhaustive state handling

When using union/sum-like state types, handle all variants explicitly.

```python
def describe_state(state: SimulationState) -> str:
    match state:
        case Prepared():
            return "prepared"
        case Running():
            return "running"
        case Completed():
            return "completed"
        case Failed():
            return "failed"
```

Use type checkers that can detect missing cases where practical.

Exhaustiveness checking is valuable because it turns future state additions into visible compile/static-check failures rather than silent runtime omissions.

---

# 7. Validation and Invariants

## 7.1 Validate as early as possible

Prefer rejecting invalid data at construction or parsing boundaries.

```python
@dataclass(frozen=True)
class MDConfig:
    timestep_fs: float

    def __post_init__(self) -> None:
        if self.timestep_fs <= 0:
            raise ValueError("timestep_fs must be positive")
```

Avoid carrying invalid values deep into the program and validating only at execution time.

---

## 7.2 Once validated, internal code may trust the type

Do not repeat the same defensive validation at every internal call site.

Preferred flow:

```text
raw input
    ↓
parse
    ↓
validate
    ↓
typed valid object
    ↓
internal computation
```

---

## 7.3 Fail fast on violated invariants

Unexpected states, impossible states, and programming errors should fail loudly.

Avoid silent fallback.

For example, unknown config keys should usually be an error rather than silently ignored.

---

# 8. Error Handling

## 8.1 Error handling depends on software level

Do not apply a single exception-vs-result rule everywhere.

### Atomic/internal functions

If a violated condition indicates a programming error or broken invariant, raising is appropriate.

```python
def calculate_energy(structure: Structure) -> float:
    if not structure.is_normalized:
        raise ValueError("structure must be normalized")
```

### Higher-level orchestration

A higher layer with enough context may catch the error and convert it into a domain-level value.

```python
def run_experiment(config: ExperimentConfig) -> ExperimentState:
    try:
        result = run_simulation(config)
    except NumericalInstabilityError as error:
        return Failed(error=error)

    return Completed(result=result)
```

Rule:

> **Errors should propagate only as far as needed to reach the nearest layer capable of making a meaningful decision.**

---

## 8.2 Expected domain failures may be values

When failure is a normal branch of the domain model, returning a value is often preferable.

Examples:

```python
Pose | None
```

or:

```python
Result[Pose, PoseNotFound]
```

Both are acceptable.

Use the richer result type when:

- there are multiple failure modes;
- the reason matters to callers;
- failure needs structured downstream handling.

Use `None` when failure is simple and self-explanatory.

---

## 8.3 Do not wrap framework errors by default

Do not mechanically translate every third-party exception into a custom application exception.

Let framework errors propagate unless:

- callers need a stable application-level API;
- the framework error leaks irrelevant implementation details;
- multiple backend errors need to be normalized;
- the higher-level error carries materially better semantics.

---

# 9. Configuration

## 9.1 Configuration is data

Complex configuration should be represented explicitly.

Prefer:

```python
@dataclass(frozen=True)
class SimulationConfig:
    temperature_kelvin: float
    duration_ns: float
    timestep_fs: float
```

over a mutable object configured through many setters.

---

## 9.2 Prefer hierarchical configuration for complex frameworks

For a complex engine such as OpenMM, separate conceptual levels.

Example:

```yaml
simulation:
  temperature_kelvin: 300
  duration_ns: 100

openmm:
  integrator: langevin_middle
  friction_per_ps: 1.0
  platform: CUDA
  precision: mixed
```

Hierarchy should reflect conceptual ownership, not merely file organization.

---

## 9.3 Scientific assumptions should be explicit

Defaults are appropriate for implementation mechanics.

Examples that may reasonably have defaults:

```python
checkpoint_interval_steps = 1000
log_format = "json"
```

Parameters that materially change scientific interpretation should generally be explicit:

```python
temperature_kelvin
ensemble
duration_ns
timestep_fs
protonation_protocol
force_field
```

Rule:

> **Defaults are for mechanics, not hidden scientific assumptions.**

---

# 10. Abstraction and Generalization

## 10.1 Prefer concrete-first development

When solving one real use case, implement that use case directly.

Prefer:

```python
run_fsp1_metadynamics(...)
```

before designing a generic engine such as:

```python
run_metadynamics(
    system,
    collective_variables,
    bias,
    scheduler,
    output_policy,
    termination_rule,
)
```

unless the generic structure already exists in a proven implementation.

---

## 10.2 Wrong abstraction is worse than duplication

A little duplication is acceptable when the common structure is not yet understood.

Refactor after repeated concrete cases reveal a stable abstraction.

Rule:

> **Abstract from evidence, not imagination.**

---

## 10.3 Extensibility is not free

Do not add factories, registries, adapters, or plugin systems merely because another backend or use case might exist someday.

Prefer the direct implementation until extension pressure becomes real.

---

# 11. Frameworks and Dependencies

## 11.1 Use standard frameworks directly

Do not wrap a mature standard framework solely for architectural purity.

Acceptable:

```python
def simulate(system: openmm.System) -> openmm.State:
    ...
```

Do not automatically introduce:

```python
MolecularSystem
SimulationBackendAdapter
BackendFactory
SimulationStateAdapter
```

unless they solve a current problem.

---

## 11.2 Framework coupling is not automatically technical debt

A direct dependency on OpenMM, PyTorch, NumPy, etc. is acceptable when the project is genuinely built on that ecosystem.

Abstract only when there is a concrete reason such as:

- backend replacement;
- testing difficulty;
- unstable API surface;
- repeated translation logic;
- a genuine domain boundary.

---

## 11.3 Prefer ecosystem maturity over aesthetic purity

If an industry-standard library is:

- stable;
- fast;
- widely used;
- well supported;

prefer using it even if:

- its API is imperfect;
- it is mutable;
- typing is incomplete.

Missing typing can often be supplemented locally with:

- type stubs;
- helper functions;
- thin wrappers only where needed.

Do not rebuild an ecosystem merely to obtain a prettier architecture.

---

# 12. Testing

## 12.1 Unit tests are foundational

Pure and atomic functions should have focused unit tests.

Test:

- edge cases;
- invariants;
- expected failure modes;
- state transitions;
- numerical behavior.

---

## 12.2 Use realistic small integration tests for orchestration

For orchestration code, prefer small realistic end-to-end or integration tests.

Example:

```text
small input structure
    ↓
real preparation
    ↓
short simulation
    ↓
result validation
```

This provides confidence that components actually work together.

---

## 12.3 Test observable behavior, not implementation structure

Mocks and fakes are primarily tools for isolating expensive or external dependencies.

Prefer:

```python
backend = FakeBackend(result=expected)

actual = run_experiment(config, backend)

assert actual == expected
```

over:

```python
mock_backend.run.assert_called_once_with(system, config)
```

unless call structure itself is part of the contract.

Avoid tests that make harmless refactoring unnecessarily difficult.

---

## 12.4 Use mocks pragmatically

Mocks are appropriate when the real dependency is:

- expensive;
- slow;
- non-deterministic;
- remote;
- hardware-dependent.

But integration tests should still cover the real boundary separately.

---

# 13. Performance

## 13.1 Optimize according to project phase

### Proof of concept

Prefer:

- readability;
- directness;
- iteration speed.

A 20% performance gain usually does not justify major complexity.

### Large-scale benchmark / production computation

Performance becomes a first-class concern.

At scale, even 20% may justify optimization.

---

## 13.2 Profile before introducing major complexity

Prefer measured bottlenecks over speculative optimization.

Performance-oriented mutation, caching, vectorization, or lower-level code are acceptable when supported by workload evidence.

---

# 14. Scientific Reproducibility

## 14.1 Reproducibility is a first-class requirement

A computational experiment should record enough information to reconstruct the procedure.

Record at minimum:

- input artifacts;
- configuration;
- code version;
- dependency/environment version;
- random seed where relevant;
- execution metadata;
- produced artifacts.

---

## 14.2 Prefer immutable experiment artifacts

Do not overwrite important scientific results in place.

Prefer:

```text
runs/
  run_001/
    config.yaml
    inputs/
    results/
    provenance.json

  run_002/
    ...
```

over modifying `run_001/results` after the fact.

---

## 14.3 Target statistical reproducibility for stochastic systems

Bitwise identity is not always realistic or scientifically necessary.

For stochastic simulation, the important standard is:

> Re-running the same experimental procedure should reproduce the same statistical or scientific conclusion.

Use multiple seeds/replicates when scientific conclusions depend on stochastic variation.

---

# 15. Naming and Readability

## 15.1 Prefer explicit names over clever brevity

Prefer:

```python
binding_energy_kcal_mol
```

over:

```python
be
```

when the longer name improves interpretation.

---

## 15.2 Function signatures should communicate intent

Prefer narrowly typed, semantically clear APIs.

```python
def load_protein(path: Path) -> Protein:
    ...
```

rather than a highly dynamic convenience function whose behavior depends on runtime inference.

---

## 15.3 Avoid clever abstractions that compress code but increase reasoning cost

Code length is not a primary optimization target.

Forty lines of opaque framework machinery are not automatically simpler than eighty linear lines.

---

# 16. Decision Rules

Use the following rules when deciding how to structure new code.

## Should this be a method?

Use a method when the operation is an intrinsic capability or property of the type.

Otherwise prefer a free function or service.

---

## Should this object be mutable?

Default to immutable.

Allow mutation when:

- lifecycle semantics genuinely require it;
- mutation is internal and controlled;
- performance materially benefits;
- an external framework requires it.

Never expose arbitrary public mutation without need.

---

## Should this become a new type?

Create a new type when it:

- prevents illegal states;
- changes valid operations;
- changes required associated data;
- improves exhaustive handling;
- clarifies a real interface boundary.

Do not create a new type merely because two floats have different names.

---

## Should I create an abstraction?

Ask:

1. Is there a concrete repeated structure?
2. Do I understand what is truly common?
3. Does the abstraction reduce reasoning cost?
4. Is the implementation cost low?
5. Does it provide useful static structure today?

If not, keep the concrete implementation.

---

## Should I wrap this framework?

Usually no.

Wrap only when the wrapper solves a current problem.

---

## Should this parameter have a default?

A default is acceptable when it is mechanical and low-risk.

Require explicit input when it materially affects scientific interpretation.

---

## Should this failure raise or return a value?

Raise when:

- an invariant is broken;
- the caller violated a contract;
- the failure is unexpected at this abstraction level.

Return a value when:

- failure is an expected domain outcome;
- the caller is expected to branch on it;
- the failure reason is part of the domain state.

A higher layer may catch an exception and convert it into a value.

---

# 17. Preferred Patterns

Prefer:

- `@dataclass(frozen=True)`
- `Protocol`
- union/sum-style state modeling
- exhaustive `match`
- explicit typed config objects
- hierarchical config files
- free functions for algorithms
- narrow capability methods
- construction-time validation
- fail-fast invariant checks
- behavior-oriented tests
- small integration tests
- concrete-first development
- direct use of mature frameworks
- explicit scientific assumptions
- immutable run artifacts
- profiling-driven optimization

---

# 18. Anti-Patterns

Avoid by default:

## Mutable bag objects

```python
obj.foo = ...
obj.bar = ...
obj.mode = ...
obj.result = ...
```

especially when arbitrary combinations are invalid.

---

## God objects

```python
molecule.generate_conformers()
molecule.dock()
molecule.calculate_partial_charge()
molecule.run_md()
molecule.save_results()
```

---

## Deep inheritance

```text
Base
  -> Generic
    -> Specialized
      -> BackendSpecific
```

---

## Premature framework abstraction

```text
BackendFactory
AdapterRegistry
ProviderResolver
PluginManager
```

when only one concrete backend exists and no present problem requires these layers.

---

## Ambiguous dynamic APIs

```python
obj = load(path)
```

when `obj` may be several unrelated types and the caller already knows which one is intended.

---

## Silent fallback

Unknown fields, malformed scientific inputs, or invalid states should not silently fall back to defaults.

---

## Semantic wrapper proliferation

Avoid creating many wrapper classes that add little behavioral or static value.

---

## Hidden scientific defaults

Avoid APIs where scientifically consequential assumptions are silently chosen.

---

# 19. Example Project Structure

A typical scientific Python project following these principles might look like:

```text
project/
├── pyproject.toml
├── src/
│   └── project/
│       ├── domain/
│       │   ├── molecule.py
│       │   ├── protein.py
│       │   └── states.py
│       ├── config/
│       │   ├── models.py
│       │   └── parsing.py
│       ├── algorithms/
│       │   ├── docking.py
│       │   ├── conformers.py
│       │   └── charges.py
│       ├── simulation/
│       │   ├── protocols.py
│       │   └── openmm.py
│       ├── workflows/
│       │   └── experiment.py
│       └── io/
│           ├── protein.py
│           └── ligand.py
├── tests/
│   ├── unit/
│   └── integration/
├── configs/
└── runs/
```

This is not a mandatory architecture.

Do not create directories or layers until the project needs them.

The structure should grow from demonstrated boundaries rather than precede them.

---

# 20. Code Review Checklist

Before merging code, ask:

- [ ] Is the behavior correct?
- [ ] Can the important control flow be understood quickly?
- [ ] Are function inputs and outputs statically clear?
- [ ] Can an invalid state be represented unnecessarily?
- [ ] Is public mutation exposed without a good reason?
- [ ] Does this method represent an intrinsic capability of the type?
- [ ] Would this algorithm be clearer as a free function?
- [ ] Is an abstraction being introduced before its common structure is known?
- [ ] Is a framework being wrapped only for architectural aesthetics?
- [ ] Are scientifically meaningful assumptions explicit?
- [ ] Is configuration represented as structured data?
- [ ] Are invariant violations detected early?
- [ ] Are expected failures represented at the appropriate software layer?
- [ ] Do tests validate observable behavior rather than incidental implementation details?
- [ ] Is there at least one realistic integration test for important orchestration?
- [ ] Is a performance optimization justified by the workload or profiling?
- [ ] Can the scientific computation be reproduced from recorded inputs/configuration/provenance?

---

# 21. Short Form

When in doubt:

> **Make illegal behavioral states hard to represent.**  
> **Keep public state stable and mutation controlled.**  
> **Use methods for intrinsic capabilities and functions for algorithms.**  
> **Prefer Protocol and composition to implementation inheritance.**  
> **Use types aggressively where they constrain behavior, not where they merely rename values.**  
> **Validate early and fail loudly.**  
> **Handle errors at the nearest layer capable of making a meaningful decision.**  
> **Prefer declarative code when it reduces reasoning cost.**  
> **Abstract only after the structure is demonstrated.**  
> **Use mature frameworks directly unless a wrapper solves a real problem.**  
> **Make scientific assumptions explicit.**  
> **Treat configuration and experiment artifacts as immutable data.**  
> **Test behavior, then integration.**  
> **Optimize when scale makes performance matter.**
