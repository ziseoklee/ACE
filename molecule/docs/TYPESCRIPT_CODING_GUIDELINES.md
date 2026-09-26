# TypeScript Coding Guidelines

## Purpose and priorities

These rules translate the project's Python programming philosophy into TypeScript. Prioritize correctness, readability, type safety, measured performance, reproducibility, explicitness, development speed, extensibility, then brevity. Apply the rules to application and scientific code; adapt framework conventions when interoperability matters.

> Make legal states and API results clear at the call site. Validate external data before trusting it. Keep state controlled, algorithms explicit, and abstractions proportional to demonstrated needs.

## 1. Compiler and project defaults

Enable `strict`, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, and `noImplicitOverride` for new projects where practical. Keep type checking in CI. Use linting for unsafe assertions and floating promises. Do not weaken a project-wide check just to accommodate one library; narrow the unsafe boundary locally.

```json
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "noImplicitOverride": true
  }
}
```

TypeScript checks types at compile time; its annotations do not validate JSON, HTTP responses, environment variables, or files at runtime. Treat such values as `unknown` until parsed.

## 2. Model real states with discriminated unions

Prefer variants whose fields express the valid state over independent flags and optional fields.

```ts
type SimulationState =
  | { readonly kind: "prepared"; readonly system: System }
  | { readonly kind: "running"; readonly jobId: string }
  | { readonly kind: "completed"; readonly result: SimulationResult }
  | { readonly kind: "failed"; readonly error: SimulationError };

function describe(state: SimulationState): string {
  switch (state.kind) {
    case "prepared":
      return "prepared";
    case "running":
      return `running: ${state.jobId}`;
    case "completed":
      return `completed: ${state.result.id}`;
    case "failed":
      return `failed: ${state.error.message}`;
    default:
      return assertNever(state);
  }
}

function assertNever(value: never): never {
  throw new Error(`Unexpected state: ${JSON.stringify(value)}`);
}
```

Adding a variant should produce a compile error at unhandled switches. Avoid `{status: string; result?: Result; error?: Error}` when it permits contradictory combinations. Use literal unions or enums for finite choices, according to the surrounding codebase. A union is most useful when variants have different data or behavior.

## 3. Prefer precise, intention-revealing APIs

When the caller knows the desired type, make the operation and return type explicit.

```ts
function loadProtein(path: string): Promise<Protein> {
  /* ... */
}
function loadLigand(path: string): Promise<Ligand> {
  /* ... */
}
```

Avoid a broad `load(path): Promise<Protein | Ligand | Trajectory>` that forces every caller to inspect a runtime result. A broad API is appropriate when discovery of the input kind is actually part of the task.

Define parameter and result types for exported APIs. Prefer `unknown` to `any` at dynamic boundaries. Narrow with validation or control flow; avoid `as T` as a substitute for validation. Avoid `!` when the absence case can be handled explicitly. Use `satisfies` for checking a literal's shape without unnecessarily widening its inferred type.

```ts
const settings = { precision: "mixed" } satisfies OpenMMSettings;
```

Type assertions are acceptable when an external API has a known typing gap and the invariant is documented and checked as close to the assertion as possible.

## 4. Immutable public data, controlled mutation

Use `readonly` for configuration and value-like objects. Prefer creating a new value to mutating a published experiment definition.

```ts
type SimulationConfig = Readonly<{
  temperatureKelvin: number;
  durationNs: number;
  backend: Readonly<{
    platform: "CPU" | "CUDA";
    precision: "single" | "mixed";
  }>;
}>;
```

`readonly` is a compile-time constraint and is shallow; it does not freeze nested values or protect against untyped JavaScript. Validate and copy external input at the boundary when ownership matters. Use `Object.freeze` only when runtime immutability is needed, understanding that a shallow freeze does not recursively freeze nested objects.

Local mutation in a loop, buffer, cache, or framework integration is fine when it improves clarity or speed. If an object must change over time, expose operations that maintain its invariants rather than public writable fields. Avoid accumulating setters for what is better represented as an immutable config value.

## 5. Data, capabilities, and algorithms

Plain objects and types work well for domain data. Use a class when identity, lifecycle, encapsulated mutable state, or an intrinsic capability makes it useful. An interface or structural type can describe a small capability even with one implementation.

```ts
interface CollectiveVariable {
  evaluate(system: System): number;
}

function generateConformers(
  molecule: Molecule,
  config: ConformerConfig,
): Conformer[] {
  /* algorithm */
}
```

Properties such as atom count and formal charge, and a molecule's canonical representation, may belong on a molecule abstraction. Conformer generation, docking, charge assignment, and simulation are separate algorithms unless the chosen ecosystem already defines a useful convention. Prefer interfaces and composition over deep implementation inheritance. An interface should clarify a real capability; avoid factories, registries, adapters, and class hierarchies built for hypothetical backends.

## 6. Validate at trust boundaries

Parse file content, HTTP payloads, URL parameters, and environment variables before passing them to typed internal code. Reject unknown fields in scientific configuration when a typo could silently change the experiment. Use a maintained runtime schema library if it fits the project; a focused hand-written parser is also fine. The important contract is runtime validation, not a particular package.

```ts
type MDConfig = Readonly<{ temperatureKelvin: number; timestepFs: number }>;

function parseMDConfig(input: unknown): MDConfig {
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    throw new Error("Expected an MD config object");
  }
  const fields = input as Record<string, unknown>;
  const keys = Object.keys(fields);
  if (keys.some((key) => key !== "temperatureKelvin" && key !== "timestepFs")) {
    throw new Error("Unknown MD config field");
  }
  const { temperatureKelvin, timestepFs } = fields;
  if (
    typeof temperatureKelvin !== "number" ||
    !Number.isFinite(temperatureKelvin) ||
    temperatureKelvin <= 0
  ) {
    throw new Error("temperatureKelvin must be a positive finite number");
  }
  if (
    typeof timestepFs !== "number" ||
    !Number.isFinite(timestepFs) ||
    timestepFs <= 0
  ) {
    throw new Error("timestepFs must be a positive finite number");
  }
  return { temperatureKelvin, timestepFs };
}
```

The internal `Record` assertion only permits inspection; the individual values remain `unknown` until checked. Avoid repeatedly validating the same invariant after a trusted parser has established it. TypeScript is structurally typed, so exported types alone cannot guarantee that all callers used a parser; keep construction boundaries clear where that guarantee matters.

## 7. Error handling by layer

Throw for broken invariants and unexpected failures inside atomic operations. Catch at the nearest layer that can make a meaningful decision. Use an explicit union for expected domain outcomes whose failure reason callers need to handle.

```ts
type DockResult =
  | { readonly ok: true; readonly pose: Pose }
  | { readonly ok: false; readonly reason: "no-pose" | "invalid-input" };
```

Use `undefined` or `null` when absence is the only meaningful outcome and its meaning is obvious. Do not convert every third-party exception into a custom class. In a `catch`, the value may be anything; inspect it before reading properties.

```ts
try {
  return await runSimulation(config);
} catch (error: unknown) {
  if (error instanceof NumericalInstabilityError) {
    return { kind: "failed", error } as const;
  }
  throw error;
}
```

Avoid empty catches, broad fallback values, and errors that silently change scientific semantics. Handle rejected promises deliberately; do not fire and forget without an explicit ownership or logging policy.

## 8. Configuration and scientific meaning

Represent complex settings as hierarchical, validated data. Separate conceptual experiment choices from backend settings when that distinction helps interpretation.

```ts
type ExperimentConfig = Readonly<{
  simulation: Readonly<{
    temperatureKelvin: number;
    durationNs: number;
    timestepFs: number;
    ensemble: "NVT" | "NPT";
  }>;
  openmm: Readonly<{
    integrator: "langevin-middle";
    platform: "CPU" | "CUDA";
    precision: "single" | "mixed";
  }>;
}>;
```

Require choices that affect scientific interpretation, such as protonation protocol, force field, ensemble, and sampling duration. Defaults are suitable for mechanics such as log formatting or checkpoint interval when documented. A config file is helpful for large experiments, but parse and validate it into typed data before execution.

## 9. Readability and control flow

Prefer clear transformations (`filter`, `map`) when they make the computation easy to follow. Use a loop when it makes error handling, early exit, or state changes clearer. Name an intermediate result when it is reused, carries domain meaning, or helps inspection; do not split every one-time expression mechanically. Avoid chains that obscure order, allocation cost, or failure behavior.

Prefer concrete, linear code for a single use case. A little duplication is cheaper than a wrong abstraction. Refactor after repeated cases reveal a stable common structure. A small interface can still be worthwhile early if it adds useful type information at low cost.

## 10. Frameworks and dependencies

Use established frameworks directly when their types and conventions fit the project. Do not wrap an entire API merely to conceal a dependency. Add a narrow helper, local declaration, or adapter only for a present problem: unsafe typing, repeated translation, testing, unstable APIs, or a genuine boundary. Keep unavoidable unsafe casts in one small, reviewed location.

Do not impose a class-based architecture on APIs that already work well with functions and plain objects. Adopt ecosystem conventions when they improve interoperability without weakening scientific correctness.

## 11. Testing and reproducibility

Test pure calculations, boundary parsing, invariants, state transitions, and important numerical edge cases. For orchestration, use small realistic integration tests. Fakes can isolate expensive or remote dependencies, but assert observable results and failure behavior rather than internal call counts unless call order is the contract.

Record inputs, validated config, code and dependency versions, seeds, execution metadata, and output artifacts. Avoid overwriting important experiment results. For stochastic work, use replicates and evaluate whether conclusions reproduce statistically; byte-for-byte equality is not the universal goal. Optimize measured bottlenecks as workload scales, while keeping exploratory code understandable.

## 12. TypeScript-specific design decisions

These decisions need more than a direct translation from Python. The defaults below are recommendations, not previously confirmed personal preferences.

### 12.1 Structural typing versus nominal identity

TypeScript checks most object compatibility by shape. Two domain concepts with the same shape can therefore be interchangeable even when the experiment says they must not be. Use ordinary structural types for capabilities and data that really are interchangeable. Consider a branded type only when accidental interchange is plausible, costly, and recurring—for example, mixing an atom index with a residue index. A brand is erased at runtime and must be created at a validated boundary.

```ts
declare const atomIndexBrand: unique symbol;
type AtomIndex = number & { readonly [atomIndexBrand]: true };

function parseAtomIndex(value: number, atomCount: number): AtomIndex {
  if (!Number.isInteger(value) || value < 0 || value >= atomCount) {
    throw new RangeError("Invalid atom index");
  }
  return value as AtomIndex; // the validated construction boundary
}
```

Do not brand every number or string. Where unit conversion is an actual source of mistakes, decide whether descriptive names, a validated unit library, or a narrowly branded type provides the least costly protection. Structural typing also does not make arbitrary object input an _exact_ schema: fresh object literal checks are not runtime unknown-key validation.

### 12.2 Absence has several meanings

Choose deliberately among `field?: T` (the property may be absent), `field: T | undefined` (the property exists, but its value may be undefined), and `field: T | null` (an explicit null value). With `exactOptionalPropertyTypes`, an optional property is more faithful to the distinction between an absent key and an explicitly undefined key. Use `??` for a fallback when `0`, `false`, or `""` are valid values; `||` uses truthiness instead. Use optional chaining only when absence is allowed by the domain, rather than masking a violated invariant.

```ts
type RunOptions = { readonly seed?: number };
const seed = options.seed ?? generateSeed(); // seed 0 remains valid
```

### 12.3 Type inference versus explicit annotations

Let the compiler infer simple local types; specify exported function contracts and scientifically meaningful result variants. Use `as const` to preserve literal values when appropriate and `satisfies` to verify a configuration literal without discarding its useful inferred type. Avoid complicated conditional or mapped types built only to save a small amount of duplication. Ask whether a colleague can understand the error message and the public signature without reverse-engineering type-level code.

Use `interface` when an extendable object capability is helpful and `type` when composing unions, tuples, mapped types, or aliases. Neither keyword is inherently more correct for a simple object. Avoid declaration merging in project-owned domain contracts unless it solves a real integration need.

### 12.4 Asynchrony is part of the contract

An async operation should have clear ownership of rejection, cancellation, and cleanup. Propagate a failure or translate it at the layer that can act on it. Expose `AbortSignal` when caller-directed cancellation is a real requirement and the underlying operation can honor it. Use `try/finally` for resources whose cleanup must occur on success, rejection, or cancellation.

Choose concurrency deliberately: `Promise.all` rejects when one task rejects; `Promise.allSettled` allows a caller to inspect every outcome. Decide whether a multi-run scientific job should fail as a unit or retain partial results. Preserve a mapping from each result to its input and seed rather than relying on completion order. Avoid unawaited promises unless the application explicitly owns their lifecycle and failure reporting.

### 12.5 Static types versus emitted JavaScript

Types, interfaces, `readonly`, and type assertions do not provide runtime validation. `import type` is erased; a runtime value must use a value import. Choose module and resolution settings for the actual runtime and build tool, then test the emitted program or bundle. A TypeScript file that type-checks but cannot resolve its runtime imports is still broken. This matters especially when publishing a package used by other environments.

When using a third-party JavaScript package with imperfect declarations, document the observed runtime contract, write a narrow adapter or declaration if useful, and integration-test that boundary. Do not let a declaration file become an unverified claim that external data is valid.

### 12.6 Collection invariants and numeric assumptions

With `noUncheckedIndexedAccess`, an array lookup may be `undefined`; handle it or establish a local invariant before using it. `ReadonlyArray<T>` makes mutation through that reference unavailable but does not recursively freeze elements. Distinguish an empty collection from a missing collection when they carry different meanings.

JavaScript `number` admits `NaN`, infinities, fractional values, and precision limits. Validate finite values, integer counts and indices, and domain ranges at input boundaries. In scientific code, document units and precision expectations; type `number` alone cannot enforce them.

### 12.7 Decisions to settle per project

The earlier conversation did not establish personal preferences on these points. Pick a policy when a real TypeScript project exposes the trade-off:

1. Which domain identifiers or units, if any, merit brands or a unit library?
2. What do `undefined`, `null`, and an omitted field mean in each external API?
3. Should a batch reject as a whole, or return individually typed outcomes?
4. Which runtime and module system will execute the emitted JavaScript?
5. Is an API public to other packages, or only internal to one application?

## 13. Common anti-patterns

- `any` or unchecked `as` throughout the codebase, especially for parsed JSON.
- Boolean flags plus optional fields that allow impossible state combinations.
- Public mutable config and domain objects whose invariants callers can bypass.
- A god object with methods for every algorithm that can operate on its data.
- Deep base-class hierarchies and speculative backend frameworks.
- Silent fallback on unknown configuration fields or missing scientific parameters.
- Generic loaders with unpredictable return unions when the caller knows the kind.
- Wrapper types for every unit or identifier without a demonstrated correctness benefit. Use unit-bearing names or a validated unit library where conversions genuinely cause errors.
- Tests that lock down private calls while missing externally observable behavior.

## 14. Review checklist

1. Can the type checker distinguish the meaningful states and require exhaustive handling?
2. Does each exported function communicate what it accepts and returns?
3. Is external input validated at runtime before it enters trusted code?
4. Can callers mutate state in a way that breaks an invariant?
5. Is this operation intrinsic to the object, or an algorithm applied to it?
6. Does this abstraction solve a current problem at an acceptable cost?
7. Are scientific assumptions explicit, and are unknown configuration fields rejected?
8. Does each error reach a layer that can actually decide what to do?
9. Do tests check outcomes and important boundaries?
10. Can another researcher reconstruct the procedure and assess statistical reproducibility?
11. Are absent, undefined, and null values distinguished where their behavior differs?
12. Are structural types interchangeable only where that is actually safe?
13. Who owns promise rejection, cancellation, cleanup, and partial batch results?
14. Have the real JavaScript runtime and module resolution been verified?

## TypeScript documentation

- [Type compatibility and structural typing](https://www.typescriptlang.org/docs/handbook/type-compatibility)
- [Narrowing and exhaustive checking](https://www.typescriptlang.org/docs/handbook/2/narrowing)
- [Exact optional property types](https://www.typescriptlang.org/tsconfig/exactOptionalPropertyTypes.html)
- [Unchecked indexed access](https://www.typescriptlang.org/tsconfig/noUncheckedIndexedAccess.html)
- [Type assertions and runtime behavior](https://www.typescriptlang.org/docs/handbook/2/everyday-types.html)
- [Choosing module compiler options](https://www.typescriptlang.org/docs/handbook/modules/guides/choosing-compiler-options.html)
