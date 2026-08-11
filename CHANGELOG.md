# Changelog

## 1.0.6 - 2026-08-11
- Reworked the webapp 3D visualization into a single unified viewer with per-task toggle buttons and max-wins residue coloring.
- Added in-view dynamic prediction intensity bars that appear per selected task and match each task color.
- Improved first-render UX by syncing initial task button active states with default active task selections.
- Added curated-input-only ground-truth loading and contextual ground-truth hints in the viewer panel.
- Added per-task model lookup and multi-task inference orchestration with merged prediction outputs.
- Updated featurizer substitution-matrix compatibility for newer Biopython versions.

## 1.0.5 - 2026-08-11
- Added a production model catalog at src/NodeCoder/model_catalog.json and locked webapp inference to catalog-registered task sets.
- Simplified webapp inference UX to task-first selection with automatic model configuration lookup.
- Replaced residue-level table preview with structure artifact outputs (PNG gallery and HTML artifact list) in the webapp.
- Added compatibility shim src/google_drive_downloader.py to satisfy torch-geometric import expectations in legacy environments.
- Improved webapp robustness by handling legacy predict-path indexing failures as non-fatal when prediction CSV is already produced.

## 1.0.4 - 2026-08-11
- Added an interactive Gradio web app at src/NodeCoder/webapp.py for protein-level prediction workflows.
- Added packaged web app entry point (NodeCoder-webapp) and optional dependency group webapp in setup metadata.
- Added web app launch instructions to README for both packaged and local module launch flows.

## 1.0.3 - 2026-08-11
- Aligned installation instructions with current PyPI behavior (`pip install NodeCoder`) instead of TestPyPI.
- Added packaged CLI usage examples (`NodeCoder-cli`) in README installation/usage section.
- Updated setup metadata homepage URL to canonical GitHub repository path and bumped package version to 1.0.2.

## 1.0.2 - 2026-08-11
- Fixed inconsistent indentation in predict module that caused syntax errors.
- Added NodeCoder CLI entry module at src/NodeCoder/main.py to restore setup.py console script target.
- Updated predict.main to accept a default protein ID so running module entry no longer raises an immediate TypeError.
