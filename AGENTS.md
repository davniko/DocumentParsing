# AGENTS.md

Operating principles for any coding agent (Claude Code, Codex, or otherwise) working on this code. These are non-negotiable defaults — they apply to every fix, implementation, refactor, and optimization unless an instruction in the same task explicitly overrides them.

If you cannot satisfy these requirements for a given task, **stop and report back**. Do not ship a degraded version.

---

## 1. Diagnose before you act

Every task begins with investigation, not implementation.

- Investigate, analyze, inspect, and diagnose before proposing or writing any change. Trace data flow end-to-end; do not pattern-match on surface symptoms.
- For performance work, profile and benchmark first to locate the actual bottleneck. Do not optimize from intuition.
- Consult official docs, API references, papers, or vetted technical write-ups when uncertain. Verifying against authoritative sources beats educated guessing.
- Do not work off assumptions. Confirm the failure mode, the root cause, and the data shape with evidence before touching code.

## 2. Confidence threshold for action

There are exactly two valid outcomes of a diagnosis pass:

- **Confident** — root cause identified with certainty, and a fix that meets every requirement in §4 is clear. → Implement immediately.
- **Not confident** — anything is uncertain, missing, ambiguous, or rests on a weak assumption. → Stop and report back per §6.2.

A confident-looking guess is not confidence. If you would hedge ("probably", "I think this should…"), you are *not confident*. Never fix a hypothesis — validate it first with a probe or repro, then fix.

## 3. What is forbidden

Never acceptable:

- **No silent fallbacks.** Errors must surface. Default values that mask broken or missing inputs are bugs, not safety nets.
- **No partial implementations.** All branches handled and tested. No stubs, no `TODO: handle X later`, no "I'll wire up the edge case in a follow-up".
- **No "legacy support" or backward-compat shims** unless explicitly requested. No dead paths "just in case".
- **No hacky, wonky, patchy, guesswork, or band-aid code.** No magic numbers without justification. No fixes based on weak assumptions. No fixes that work today and rot tomorrow.
- **No sweeping problems under the rug.** Stopping the error message without addressing the cause is worse than no fix.
- **No silent regressions** in data, output quality, or performance (see §4).
- **No dead code, commented-out blocks, or leftover debug scaffolding** in shipped changes.

If a clean fix is not possible inside the agreed constraints, stop and report (§6.2) instead of shipping something dirty.

## 4. Quality bar

Every change must be **clean, correct, AND optimal** — all three. Trade-offs are flagged in the report, not silently chosen.

### 4.1 Correctness and data fidelity

- Output must be the exact required data with the exact required quality: schema, types, units, ranges, ordering, and semantic meaning all preserved.
- **Zero regressions** in expected or required data, output, or quality.
- **Zero unaddressed side effects.** If a change touches anything outside its stated scope, that impact is identified, evaluated, and either eliminated or explicitly justified in the report.

### 4.2 Performance

Performance is a first-class requirement, not an afterthought.

- After any change, processing must be **at least as fast** as before on the relevant workload. Faster is the default goal.
- If a feature expansion forces a non-negligible perf hit, optimize until parity is restored or surpassed. A negligible hit is acceptable; an unjustified non-negligible hit is not.
- Code should be **fast, multithreaded where appropriate, memory-efficient, and safe** (thread-safety, no data races, no UB).
- Track memory actively — avoid OOMs, runaway allocations, and leaks. Streaming and chunked processing beat materializing large intermediates.
- No new blocking calls in hot paths without an explicit reason and a measurement showing throughput is preserved.

### 4.3 Design and structure

Correctness and speed are not enough — the result must be well-shaped.

- **Nail the right abstraction.** Pick the level of generality that fits the actual problem — not narrower (forces duplication later), not wider (premature generalization, dead flexibility, harder to reason about). When in doubt, err toward the concrete and let real second and third use cases drive the abstraction.
- **Design the interface deliberately.** A public surface is a contract: minimal, orthogonal, hard to misuse. Inputs, outputs, and ownership are explicit. Side effects are named. Do not expose internals just because it is convenient at the call site.
- **Structure with reason.** Module, file, and function boundaries follow the shape of the problem, not the order code was written. Related things live together; unrelated things do not. If a module is hard to name, it is probably wrong.
- **Build for extension only where extension is real.** Add hooks and seams when a concrete second case is driving them, not on speculation. Speculative flexibility is forbidden under §3 — over-generalization is its own form of mess.
- **Be consistent with the surrounding codebase.** Match existing patterns unless there is a specific, justified reason to diverge — and if you diverge, say so in the report.
- Naming is precise. Comments explain *why*, not *what*. Code is idiomatic for the language and framework.

## 5. Validate every change — test, probe, benchmark

A change is not done until it is *proven* done.

- **Test.** Write or extend targeted tests covering the changed behavior, including the failure mode that motivated the change. Re-run the relevant suite and report the result.
- **Validate manually with probes.** Instrument the code path or run a targeted probe to confirm the fix actually triggers and resolves the symptom under realistic conditions. Don't assume — observe.
- **Benchmark.** Measure before and after. Report concrete deltas in latency, throughput, and memory. For perf passes, benchmark every component touched.
- **Make the fix airtight.** Confirm the issue cannot recur through the same path. If it could, harden the path or add a regression guard.

If you cannot run tests/benchmarks (no environment, missing data, sandbox limit), say so explicitly — do not silently skip validation and call it done.

## 6. Reporting

The report is how the work is verified without re-doing the diagnosis. It is a deliverable, not an afterthought.

### 6.1 When you implement

- **Diagnosis.** Root cause, evidence, and how it was confirmed (probe output, profiling result, doc citation).
- **What changed.** File-by-file or component-by-component, with rationale per decision.
- **Measured impact.** Perf and memory deltas with concrete numbers. Test results. Probe outcomes.
- **Side effects considered.** What else this could touch, and why those concerns are resolved (or, if any remain, what they are).

For larger refactors, walk through every component touched: what was wrong, what changed and why, and the measured improvement.

### 6.2 When you stop instead of implementing

- State precisely what was investigated and what was learned.
- State the specific blocker — missing data, ambiguous spec, unverifiable assumption, doc you couldn't confirm, decision you can't make alone, environment limitation.
- If you have options, propose them ranked, with trade-offs. **Do not pick one and run.**

## 7. Decision flow

```
Task arrives
  └─ Investigate → diagnose → confirm root cause with evidence
       │
       ├─ Confident in cause AND in a clean+correct+optimal fix?
       │     ├─ YES → implement → test → probe → benchmark → report (§6.1)
       │     └─ NO  → STOP. Report findings, blockers, options (§6.2).
       │
       └─ NEVER ship: partial, hacky, fallback-laden, untested, unbenchmarked,
                       or quality/performance-regressing changes.
```
