# Changelog

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
