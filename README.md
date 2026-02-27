# neon-heart-game

Game built with Python (Flask) and JavaScript.

## NeuroVision AI pipeline
This repository now includes a modular reference implementation for the **NEUROVISION AI** pipeline.

- API endpoint: `POST /api/neurovision/run`
- Input: `multipart/form-data` with field `image`
- Output: structured JSON with detection, semantic understanding, explainability, activations, and 3D network state.

See `docs_neurovision_spec.md` for the full architecture specification.
