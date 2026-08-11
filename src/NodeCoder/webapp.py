import glob
import json
import os
import sys
import traceback
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd

from NodeCoder.utilities.parser import parameter_parser

AVAILABLE_TASKS = [
    "y_CHAIN",
    "y_TRANSMEM",
    "y_MOD_RES",
    "y_ACT_SITE",
    "y_NP_BIND",
    "y_LIPID",
    "y_CARBOHYD",
    "y_DISULFID",
    "y_VARIANT",
    "y_Artifact",
    "y_Peptide",
    "y_Nucleic",
    "y_Inorganic",
    "y_Cofactor",
    "y_Ligand",
]

TASK_LABELS = {
    "y_Ligand": "Ligand binding site",
    "y_Peptide": "Peptide binding site",
    "y_Nucleic": "Nucleic binding site",
    "y_Inorganic": "Metal ion binding site",
    "y_TRANSMEM": "Transmembrane domain",
    "y_MOD_RES": "Modified residue",
    "y_ACT_SITE": "Active site",
    "y_NP_BIND": "Nucleotide binding site",
    "y_LIPID": "Lipid binding site",
    "y_CARBOHYD": "Carbohydrate binding site",
    "y_DISULFID": "Disulfide bond",
    "y_VARIANT": "Variant site",
    "y_Artifact": "Artifact",
    "y_Cofactor": "Cofactor binding site",
    "y_CHAIN": "Protein chain",
}

MODEL_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "model_catalog.json")


def _load_model_catalog() -> Dict[str, Any]:
    if not os.path.exists(MODEL_CATALOG_PATH):
        raise ValueError(f"Model catalog not found: {MODEL_CATALOG_PATH}")

    with open(MODEL_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    if "models" not in catalog or not isinstance(catalog["models"], list):
        raise ValueError("Model catalog is invalid: expected a 'models' list.")

    return catalog


def _find_catalog_model(selected_tasks: List[str]) -> Dict[str, Any]:
    selected_key = sorted(selected_tasks)
    catalog = _load_model_catalog()

    for model in catalog["models"]:
        if not model.get("active", True):
            continue
        model_tasks = model.get("tasks", [])
        if sorted(model_tasks) == selected_key:
            return model

    available = [", ".join(m.get("tasks", [])) for m in catalog["models"] if m.get("active", True)]
    available_text = " | ".join(available) if available else "none"
    raise ValueError(f"No production model found for selected task set. Available active task sets: {available_text}")

CSS = """
:root {
  --ink: #1f2933;
  --muted: #52606d;
  --accent: #0f766e;
  --accent-2: #0369a1;
  --paper: #f8fafc;
  --panel: #ffffff;
}
.gradio-container {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif !important;
  background: radial-gradient(1200px 500px at 20% -10%, #ccfbf1 0%, rgba(204, 251, 241, 0) 60%),
              radial-gradient(1000px 600px at 90% 0%, #dbeafe 0%, rgba(219, 234, 254, 0) 55%),
              var(--paper);
}
.hero-title {
  font-size: 2.0rem;
  font-weight: 800;
  line-height: 1.2;
  color: var(--ink);
  margin-bottom: 0.2rem;
}
.hero-sub {
  font-size: 1rem;
  color: var(--muted);
  margin-bottom: 0.8rem;
}
.kpi-card {
  border: 1px solid #d9e2ec;
  border-radius: 14px;
  padding: 0.8rem;
  background: var(--panel);
}
"""


@contextmanager
def _sanitized_argv() -> None:
    original = sys.argv[:]
    try:
        sys.argv = [original[0]]
        yield
    finally:
        sys.argv = original


def _resolve_prediction_artifacts(
    protein_id: str,
    threshold_dist: int,
    trained_model_fold_number: int,
    multi_task_learning: bool,
    tasks: List[str],
    centrality_feature: bool,
    cross_validation_fold_number: int,
    epochs: int,
    performance_step: int,
    checkpoint_step: int,
    learning_rate: float,
    network_layers: List[int],
    train_ratio: float,
    train_cluster_number: int,
    validation_cluster_number: int,
) -> Tuple[str, List[str]]:
    with _sanitized_argv():
        args = parameter_parser(
            NodeCoder_usage="predict",
            protein_ID=protein_id,
            threshold_dist=threshold_dist,
            trained_model_fold_number=trained_model_fold_number,
            multi_task_learning=multi_task_learning,
            Task=tasks,
            centrality_feature=centrality_feature,
            cross_validation_fold_number=cross_validation_fold_number,
            epochs=epochs,
            performance_step=performance_step,
            checkpoint_step=checkpoint_step,
            learning_rate=learning_rate,
            network_layers=network_layers,
            train_ratio=train_ratio,
            train_cluster_number=train_cluster_number,
            validation_cluster_number=validation_cluster_number,
        )

    files = sorted(glob.glob(os.path.join(args.path_protein_results, f"{protein_id}_*")))
    return args.protein_prediction_fileName, files


def run_prediction(
    protein_id: str,
    tasks: List[str],
) -> Tuple[str, str, List[str], List[str], List[str]]:
    protein_id = (protein_id or "").strip()
    if not protein_id:
        return "Please provide a protein ID (example: KI3L1_HUMAN).", "", [], [], []

    if not tasks:
        return "Please select at least one task.", "", [], [], []

    try:
        model = _find_catalog_model(tasks)
    except ValueError:
        return "No production model is registered for the selected task set. Update src/NodeCoder/model_catalog.json.", "", [], [], []

    catalog_tasks = model["tasks"]
    threshold_dist = int(model["threshold_dist"])
    cross_validation_fold_number = int(model["cross_validation_fold_number"])
    trained_model_fold_number = int(model["trained_model_fold_number"])
    epochs = int(model["epochs"])
    learning_rate = float(model["learning_rate"])
    parsed_layers = [int(v) for v in model["network_layers"]]
    centrality_feature = bool(model.get("centrality_feature", True))
    train_ratio = float(model.get("train_ratio", 0.8))
    train_cluster_number = int(model.get("train_cluster_number", 1))
    validation_cluster_number = int(model.get("validation_cluster_number", 1))

    multi_task_learning = len(catalog_tasks) > 1

    try:
        from NodeCoder import predict
    except ImportError:
        return "PyTorch/ML dependencies not installed. Install with: pip install torch torchvision torch-geometric", "", [], [], []

    prediction_warning = ""
    with _sanitized_argv():
        try:
            predict.main(
                protein_ID=protein_id,
                threshold_dist=threshold_dist,
                trained_model_fold_number=trained_model_fold_number,
                multi_task_learning=multi_task_learning,
                Task=catalog_tasks,
                centrality_feature=centrality_feature,
                cross_validation_fold_number=cross_validation_fold_number,
                epochs=epochs,
                performance_step=50,
                checkpoint_step=50,
                learning_rate=learning_rate,
                network_layers=parsed_layers,
                train_ratio=train_ratio,
                train_cluster_number=train_cluster_number,
                validation_cluster_number=validation_cluster_number,
            )
        except IndexError:
            # Known legacy behavior: predict.main can fail after CSV prediction is written
            # while trying to save per-protein metrics using a fold index mismatch.
            prediction_warning = " Warning: per-protein metrics export was skipped due to legacy fold indexing."
        except Exception:
            raise

    prediction_csv, artifacts = _resolve_prediction_artifacts(
        protein_id=protein_id,
        threshold_dist=threshold_dist,
        trained_model_fold_number=trained_model_fold_number,
        multi_task_learning=multi_task_learning,
        tasks=catalog_tasks,
        centrality_feature=centrality_feature,
        cross_validation_fold_number=cross_validation_fold_number,
        epochs=epochs,
        performance_step=50,
        checkpoint_step=50,
        learning_rate=learning_rate,
        network_layers=parsed_layers,
        train_ratio=train_ratio,
        train_cluster_number=train_cluster_number,
        validation_cluster_number=validation_cluster_number,
    )

    if not os.path.exists(prediction_csv):
        return (
            "Prediction ran but output CSV was not found. Verify trained checkpoints and input protein availability.",
            prediction_csv,
            artifacts,
            [],
            [],
        )

    df = pd.read_csv(prediction_csv)
    protein_result_dir = os.path.dirname(prediction_csv)
    png_files = sorted(glob.glob(os.path.join(protein_result_dir, "*.png")))
    html_files = sorted(glob.glob(os.path.join(protein_result_dir, "*.html")))

    if not png_files and not html_files:
        prediction_warning += (
            " Note: structure visualization artifacts were not found. "
            "Prediction CSV and files were generated; if needed, configure structure-file inputs in the preprocessing/inference pipeline."
        )

    summary = (
        f"Prediction finished for {protein_id}. "
        f"Rows: {len(df)} | Tasks: {', '.join(catalog_tasks)} | "
        f"Catalog Model: {model['id']} | Fold: {trained_model_fold_number} | Output: {prediction_csv}"
        f"{prediction_warning}"
    )
    return summary, prediction_csv, artifacts, png_files, html_files


def build_app() -> gr.Blocks:
    with gr.Blocks(title="NodeCoder - Protein ActiveSite Predictor", css=CSS) as app:
        gr.HTML("<div class='hero-title'>NodeCoder - Protein ActiveSite Predictor</div>")
        gr.HTML(
            "<div class='hero-sub'>Predict functional and active sites from AlphaFold protein structures "
            "using your trained NodeCoder checkpoints.</div>"
        )

        with gr.Row():
            with gr.Column(scale=2):
                protein_id = gr.Textbox(
                    label="Protein ID",
                    value="KI3L1_HUMAN",
                    placeholder="Example: EGFR_HUMAN",
                )
                task_choices = [(TASK_LABELS[t], t) for t in AVAILABLE_TASKS]
                tasks = gr.CheckboxGroup(
                    choices=task_choices,
                    value=["y_Ligand"],
                    label="Prediction Tasks",
                )
                run_btn = gr.Button("Run Prediction", variant="primary")
            with gr.Column(scale=1):
                gr.HTML("<div class='kpi-card'><b>Tips</b><br/>1. Choose one or more tasks.<br/>2. The app uses the locked production model catalog in src/NodeCoder/model_catalog.json.<br/>3. Outputs are written under results/graph_*A.</div>")

        status = gr.Textbox(label="Run Status")
        output_csv = gr.Textbox(label="Prediction CSV Path")
        artifacts = gr.Files(label="Generated Artifact Files")
        structure_images = gr.Gallery(label="Structure Visualization (PNG)", columns=2, height=420)
        structure_html = gr.Files(label="Interactive Structure Views (HTML)")

        run_btn.click(
            fn=run_prediction,
            inputs=[
                protein_id,
                tasks,
            ],
            outputs=[status, output_csv, artifacts, structure_images, structure_html],
        )

    return app


def main() -> None:
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
