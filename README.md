# Risk-Budgeted Inventory-Aware Market Making

Research code and frozen validation artifacts for constrained
reinforcement learning in multi-level market making.

## Experiments

1. Fixed-penalty K4 baseline.
2. Order-aware adaptive-dual controller D2.
3. Fill-aware anti-windup residual-dual controller D3.

## Repository structure

- notebooks: complete experimental notebooks;
- src/mmrl: environment, simulator, reward, risk and TD3 code;
- configs: sanitized configuration and frozen protocols;
- results: aggregate tables, inference and figures.

Raw data, checkpoints, replay buffers and transition-level
Parquet files are intentionally excluded.

The reported experiments use frozen validation windows.
The untouched test/OOS split was not accessed.

Paper/preprint: ADD_LINK_HERE
