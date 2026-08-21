# Repository Agent Contract

## Mission

Own commodity time-series prediction research for this repository. Produce reproducible empirical comparisons from frozen datasets/splits and explicit benchmarks without treating competition scores or backtests as live trading evidence.

## Canonical authority

- Preserve dataset/revision identity, feature availability time, train/validation/test boundaries, target definition, benchmark, metric, model/config version and run provenance.
- Use external commodity/market observations from their owning canonical source where possible instead of creating a second finance-wide data authority.
- Keep observed inputs, engineered features, predictions, validation/test metrics and economic interpretation distinct.

## Autonomous execution

1. Inspect current `main`, README, open Issues/PRs, dataset/split manifests, experiment code/results, workflows and tests.
2. Continue one canonical research workline before adding another model, dataset, branch or Issue.
3. Prefer completion of a reproducible held-out experiment, leakage/baseline correction, robustness/falsification, or removal of superseded experiment code.
4. Predeclare split, metric and benchmark before interpreting new model results.
5. Bind evidence to exact code/data revisions and run the smallest relevant deterministic/evaluation checks before merge.
6. Stop at the fixed point; do not run broad model/hyperparameter sweeps without a bounded research question.

## Branch lifecycle

- Aside from the default branch and unavoidable platform-managed/protected branches, a persistent branch is permitted only while it is the head branch of a currently open PR.
- Creating a work branch creates an obligation to open or reuse its canonical PR immediately; do not use branches as backlog, continuation state, backup, archive, or evidence storage.
- After a PR is merged or closed, delete its head branch after verifying PR/main state. A branch with no open PR is an orphan and must be deleted.
- Before and after work, compare repository branches with open PR heads. Do not report cleanup/fixed point while an orphan task branch remains.
- If the available tool cannot delete a branch, record that as a tooling blocker and do not claim cleanup complete. Never create another orphan branch as a workaround.

## Merge and release are separate

### PR merge conditions

A PR may merge when the repository-local research contract is correct on the exact head revision: frozen data/split/benchmark definitions are fixed, deterministic/evaluation checks pass, result artifacts are reproducible where affected, and no unresolved review or correctness blocker remains.

A future competition score, live market observation, production inference, public deployment, or realized trading result is **not** a merge condition unless the PR specifically changes the release mechanism and pre-merge validation belongs to that bounded change.

### Research/model release conditions

Release is a separate post-merge decision. Treat a commodity prediction result/model as released only after the merged `main` revision is read back and the release artifact/surface in scope is actually verified, including exact dataset/model revision, persisted held-out evaluation artifact, published model/API/UI when applicable, deployment identity, and rollback/rebuild path.

A merged PR does not prove live predictive or trading performance. A release/data/runtime blocker may block release without invalidating a correctly merged repository change. Report merge and release independently.

## Boundaries

- Competition validation/public scores are not necessarily live out-of-sample trading results.
- Do not infer missing prices, costs, liquidity, market impact or future performance.
- Never execute commodity trades, derivatives, transfers or account actions.
- Unexecuted experiments, CI and realized market outcomes remain unverified.

## Completion report

Report empirical result Before -> After, exact dataset/split/model/benchmark evidence, Issue/PR/commit/check artifact, then report `merged` and `released` separately with direct evidence for each. Include branch cleanup state, complexity removed and the remaining blocker.