# Topological Maps Are For

This repository contains an older, compact version of NeuralSheet: a PyTorch simulation of cortical self-organisation inspired by GCAL-style recurrent settling, Hebbian plasticity, homeostasis, and wiring-efficiency analysis.

## What The Code Does

`neuralsheet.py` defines the main `NeuralSheet` model. A sheet of units receives local image patches, recurrent activity settles for several iterations, and Hebbian updates reshape the afferent and lateral weights. The L4 path is the default analysis target; the model can also run an L2/3 path for experiments with local and global recurrent interactions.

The standard workflow is:

1. Sample natural-image crops from `input_stimuli/`.
2. Train a `NeuralSheet` with recurrent settling and Hebbian plasticity.
3. Inspect receptive fields, orientation maps, phase maps, activity patterns, and recurrent connectivity.
4. Evaluate reconstruction accuracy, robustness to noise, PCA dimensionality, and UMAP structure.
5. Save generated figures and simulation tensors outside git-tracked source files.

## Repository Layout

- `neuralsheet.py`: core model and recurrent/Hebbian dynamics.
- `helpers/`: reusable plotting, data, PCA/UMAP, grating, wiring, and notebook helper code.
- `stats_collector.py`: long-running L4 statistics and noise robustness sweeps.
- `parameter_search.py`: parameter sweep script; outputs go to `parameter_search_data/`.
- `wiring_efficiency.ipynb`: main L4 analysis notebook.
- `l3_analysis.ipynb`: L3 analysis notebook split out from the L4 notebook.
- `audio/`: auditory-sheet experiments, intentionally ignored by git.
- `figures/`: generated plots, intentionally ignored by git.
- `data_l4/` and `data_l3/`: generated simulation outputs, intentionally ignored by git.

## Running Small Checks

For code edits, use lightweight validation rather than launching new sweeps:

```bash
python -m py_compile neuralsheet.py stats_collector.py parameter_search.py helpers/*.py
python - <<'PY'
import json
for path in ['wiring_efficiency.ipynb', 'l3_analysis.ipynb', 'cortical_map_demo.ipynb']:
    json.load(open(path))
print('notebook json ok')
PY
```

Large collector runs are launched explicitly through `run_stats_collector_detached.py`; do not start them just to verify imports.

## Git Policy

Git tracks root-level code/notebooks, `README.md`, `AGENTS.md`, and the Python helper modules in `helpers/`. Generated data, figures, audio experiments, input images, logs, checkpoints, and caches are ignored.
