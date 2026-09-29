# GGA on UJI cross-floor localization

Independent research branch for Wi-Fi RSSI → 2D coordinate regression. The task is
**leave one floor out within a building**. The held-out floor's official
`validationData.csv` rows are the primary final test. A separate 10-position,
3-scans-per-position, 50-step target adaptation study uses only that floor's
`trainingData.csv` support rows.

| Input | Ordinary training | GGA early search |
|---|---|---|
| Fixed 520 AP slots | `erm_mlp` | `gga_mlp` |
| Detected AP set with masked Set Transformer blocks | `erm_set` | `gga_set` |

All four methods use the same source floors, batches, support IDs, optimizer
settings and fixed 5000-step budget. GGA changes only the early source update;
Set Transformer changes only input representation. No validation label is used
to train or choose a checkpoint.

`configs/v1.json` is a three-floor implementation pilot. The predeclared
primary study `configs/full13.json` holds out all 13 floors in turn.
`docs/PROTOCOL.md` defines the experiment, `docs/PROVENANCE.md` maps adapted
code to the original repositories, and `docs/VALIDATION_AUDIT.md` quantifies
the official test's time, user, phone and spatial sampling changes.

Run one cell from the project root:

```bash
python -m scripts.run --config configs/full13.json --target B0F3 --method gga_mlp --seed 0 --output outputs/full13
```

Or run one encoder's grid on a selected GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_queue --kind mlp --config configs/full13.json --output outputs/full13
```

After every cell is complete:

```bash
python -m scripts.summarize configs/full13.json outputs/full13
```

GGA's [updated paper](https://arxiv.org/html/2502.20162v7) and
[official code](https://github.com/aristotelisballas/GGA) concern image
classification. This project adapts its optimizer to scaled coordinate
regression, and adapts the [official Set Transformer
blocks](https://github.com/juho-lee/set_transformer) for masked AP sets. It is
not a numerical reproduction of GGA's CVPR image benchmarks.
