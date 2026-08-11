import glob
import os
import re
import sys
from contextlib import contextmanager
from typing import List, Tuple

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


def _parse_model_spec(model_spec: str) -> Tuple[int, int, int, float, List[int]]:
    threshold_match = re.search(r"graph_(\d+)A", model_spec)
    folds_match = re.search(r"/(\d+)FoldCV/", model_spec)
    task_match = re.search(r"/(y_[^/]+)_HiddenLayers_", model_spec)
    layers_match = re.search(r"HiddenLayers_([0-9_]+)_", model_spec)
    epochs_match = re.search(r"_(\d+)Epochs_", model_spec)
    lr_match = re.search(r"_LR([0-9.]+)$", model_spec)

    if not (threshold_match and folds_match and task_match and layers_match and epochs_match and lr_match):
        raise ValueError("Model path format is not recognized.")

    threshold_dist = int(threshold_match.group(1))
    cross_validation_fold_number = int(folds_match.group(1))
    task = task_match.group(1)
    network_layers = [int(x) for x in layers_match.group(1).split("_") if x]
    epochs = int(epochs_match.group(1))
    learning_rate = float(lr_match.group(1))
    return threshold_dist, cross_validation_fold_number, epochs, learning_rate, network_layers


def _discover_candidate_models(tasks: List[str]) -> List[str]:
    if len(tasks) == 1:
        pattern = f"./results/graph_*A/*/*FoldCV/{tasks[0]}_HiddenLayers_*_Epochs_LR*"
    else:
        pattern = f"./results/graph_*A/*/*FoldCV/{len(tasks)}Targets_HiddenLayers_*_Epochs_LR*"

    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return [d for d in candidates if os.path.isdir(d)]


def _resolve_trained_fold(model_dir: str) -> int:
    fold_dirs = sorted(glob.glob(os.path.join(model_dir, "CheckPoints", "Fold*")))
    for fold_dir in fold_dirs:
        fold_match = re.search(r"Fold(\d+)$", fold_dir)
        if fold_match:
            return int(fold_match.group(1))
    return 1


def _auto_select_model(tasks: List[str]) -> Tuple[str, int]:
    candidates = _discover_candidate_models(tasks)
    if not candidates:
        raise ValueError("No matching trained model was found under results/ for the selected task configuration.")
    model_dir = candidates[0]
    fold_number = _resolve_trained_fold(model_dir)
    return model_dir, fold_number

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
) -> Tuple[str, pd.DataFrame, str, List[str]]:
    protein_id = (protein_id or "").strip()
    if not protein_id:
        return "Please provide a protein ID (example: KI3L1_HUMAN).", pd.DataFrame(), "", []

    if not tasks:
        return "Please select at least one task.", pd.DataFrame(), "", []

    try:
        model_spec, trained_model_fold_number = _auto_select_model(tasks)
        threshold_dist, cross_validation_fold_number, epochs, learning_rate, parsed_layers = _parse_model_spec(model_spec)
    except ValueError:
        return "No matching trained model found for selected task(s). Train a model first and try again.", pd.DataFrame(), "", []

    multi_task_learning = len(tasks) > 1

    try:
        from NodeCoder import predict
    except ImportError:
        return "PyTorch/ML dependencies not installed. Install with: pip install torch torchvision torch-geometric", pd.DataFrame(), "", []

    with _sanitized_argv():
        predict.main(
            protein_ID=protein_id,
            threshold_dist=threshold_dist,
            trained_model_fold_number=trained_model_fold_number,
            multi_task_learning=multi_task_learning,
            Task=tasks,
            centrality_feature=True,
            cross_validation_fold_number=cross_validation_fold_number,
            epochs=epochs,
            performance_step=50,
            checkpoint_step=50,
            learning_rate=learning_rate,
            network_layers=parsed_layers,
            train_ratio=0.8,
            train_cluster_number=1,
            validation_cluster_number=1,
        )

    prediction_csv, artifacts = _resolve_prediction_artifacts(
        protein_id=protein_id,
        threshold_dist=threshold_dist,
        trained_model_fold_number=trained_model_fold_number,
        multi_task_learning=multi_task_learning,
        tasks=tasks,
        centrality_feature=True,
        cross_validation_fold_number=cross_validation_fold_number,
        epochs=epochs,
        performance_step=50,
        checkpoint_step=50,
        learning_rate=learning_rate,
        network_layers=parsed_layers,
        train_ratio=0.8,
        train_cluster_number=1,
        validation_cluster_number=1,
    )

    if not os.path.exists(prediction_csv):
        return (
            "Prediction ran but output CSV was not found. Verify trained checkpoints and input protein availability.",
            pd.DataFrame(),
            prediction_csv,
            artifacts,
        )

    df = pd.read_csv(prediction_csv)
    summary = (
        f"Prediction finished for {protein_id}. "
        f"Rows: {len(df)} | Tasks: {', '.join(tasks)} | "
        f"Model: {model_spec.split('./results/')[-1]} | Fold: {trained_model_fold_number} | Output: {prediction_csv}"
    )
    return summary, df.head(300), prediction_csv, artifacts


def build_app() -> gr.Blocks:
    with gr.Blocks(title="NodeCoder - Protein ActiveSite Predictor") as app:
        gr.HTML("<div class='hero-title'>NodeCoder - Protein ActiveSite Predictor</div>")
        gr.HTML(
            "<div class='hero-sub'>Predict functional and active residue-level sites from AlphaFold protein structures "
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
                gr.HTML("<div class='kpi-card'><b>Tips</b><br/>1. Choose one or more tasks.<br/>2. The app auto-selects the newest matching trained model from results/.<br/>3. Outputs are written under results/graph_*A.</div>")

        status = gr.Textbox(label="Run Status")
        preview = gr.Dataframe(label="Prediction Preview (first 300 rows)", wrap=True)
        output_csv = gr.Textbox(label="Prediction CSV Path")
        artifacts = gr.Files(label="Generated Artifact Files")

        run_btn.click(
            fn=run_prediction,
            inputs=[
                protein_id,
                tasks,
            ],
            outputs=[status, preview, output_csv, artifacts],
        )

    return app


def main() -> None:
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)


if __name__ == "__main__":
    main()
