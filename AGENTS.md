# GGA-UJI-CrossFloor

- This orphan branch is an independent cross-floor DG/DA study. Do not modify other Femloc branches or reuse their task framework.
- `docs/PROTOCOL.md` fixes data access and the experiment meaning. `validationData.csv` is final test data only; do not select models or parameters using it.
- Use the updated GGA v7 paper and official code commit recorded in `docs/PROVENANCE.md`. Preserve its early parameter search and domain-gradient criterion when adapting cross-entropy to coordinate regression.
- Keep DG (no target training rows) separate from DA (fixed target support rows). Neither may use official validation labels to fit or select a model.
- Compare ERM and GGA with the same encoder, batches, seed, data normalization, and training steps. Treat Set Transformer as an architectural factor, not as a GGA improvement by itself.
- Only implement currently used experiments, short functional comments, source-only checks for leakage and tensor shapes. No compatibility framework or speculative fallbacks.
