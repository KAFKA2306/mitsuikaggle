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
5. Bind evidence to exact code/data revisions and run the smallest relevant deterministic/evaluation checks.
6. Stop at the fixed point; do not run broad model/hyperparameter sweeps without a bounded research question.

## Boundaries

- Competition validation/public scores are not necessarily live out-of-sample trading results.
- Do not infer missing prices, costs, liquidity, market impact or future performance.
- Never execute commodity trades, derivatives, transfers or account actions.
- Unexecuted experiments, CI and realized market outcomes remain unverified.

## Completion report

Report empirical result Before -> After, exact dataset/split/model/benchmark evidence, Issue/PR/commit/check artifact when applicable, complexity removed, and the remaining blocker.