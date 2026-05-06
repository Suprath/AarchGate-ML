# AarchGate-ML Architecture & Progress

## Project Overview
AarchGate-ML is a high-performance inference accelerator for XGBoost models, leveraging the AarchGate bit-sliced circuit engine.

## Architectural Decisions
- **Docker-First Environment**: All builds and executions are containerized to ensure consistency with AarchGate's ARM64 optimizations.
- **Fixed-Point Quantization**: Float values from XGBoost are scaled by $10^6$ and converted to `uint64_t` for bit-sliced processing.
- **Bit-Sliced Summation**: Tree results are summed using a tree of `apex.builder.Add` operations to maintain bit-plane efficiency.

## Progress Log
- [2026-05-04] Initialized project structure and implementation plan.
- [2026-05-04] Identified AarchGate submodule and its Docker build environment.
- [2026-05-04] Completed XGBoost to AarchGate converter and summation layer.
- [2026-05-04] Achieved 1524x inference speedup on 10M rows in NYC taxi benchmark.
- [2026-05-04] Modified `aarchgate_pybind.cpp` to expose `apex::builder` IR construction functions to Python, enabling dynamic model conversion.
- [2026-05-05] Debugged and stabilized JIT comparison logic; resolved Row 0 accuracy divergence.
- [2026-05-05] Implemented SIMD-accelerated lexicographical comparison (Pass 1) with 64-bit mask hardening.
- [2026-05-05] Fixed register clobbering in `SELECT` node multiplexing and optimized `ADD/SUB` passes for topological correctness.
- [2026-05-05] Achieved record throughput of 268M rows/sec (12,440x speedup) on NYC taxi benchmark.
- [2026-05-06] Reconstructed JIT compiler to run a unified, single-pass post-order topological compilation loop over the analytical IR nodes, completely eliminating compilation dependency bugs.
- [2026-05-06] Supported non-zero else branches of SELECT nodes in the hybrid popcount aggregator algebra to resolve accuracy divergences on manually structured C++ selector configurations.
- [2026-05-06] Resolved python dynamic loading missing-symbol errors by implementing `execute_vector` at the C++ level to enable seamless, zero-copy batch prediction retrieval in python.
- [2026-05-06] Achieved 100% correctness across standard C++ unit/infrastructure tests and real-world high-dimensional python ML models.

---

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
