import glob
import os
import sys
from contextlib import contextmanager
from typing import List, Tuple

import gradio as gr
import pandas as pd

from NodeCoder import predict
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
    threshold_dist: int,
    trained_model_fold_number: int,
    centrality_feature: bool,
    cross_validation_fold_number: int,
    epochs: int,
    learning_rate: float,
    network_layers: str,
    train_cluster_number: int,
    validation_cluster_number: int,
) -> Tuple[str, pd.DataFrame, str, List[str]]:
    protein_id = (protein_id or "").strip()
    if not protein_id:
        return "Please provide a protein ID (example: KI3L1_HUMAN).", pd.DataFrame(), "", []

    if not tasks:
        return "Please select at least one task.", pd.DataFrame(), "", []

    try:
        parsed_layers = [int(x.strip()) for x in network_layers.split(",") if x.strip()]
    except ValueError:
        return "Network layers must be comma-separated integers, e.g. 38,28,18,8.", pd.DataFrame(), "", []

    if not parsed_layers:
        return "Network layers cannot be empty.", pd.DataFrame(), "", []

    multi_task_learning = len(tasks) > 1

    with _sanitized_argv():
        predict.main(
            protein_ID=protein_id,
            threshold_dist=threshold_dist,
            trained_model_fold_number=trained_model_fold_number,
            multi_task_learning=multi_task_learning,
            Task=tasks,
            centrality_feature=centrality_feature,
            cross_validation_fold_number=cross_validation_fold_number,
            epochs=epochs,
            performance_step=50,
            checkpoint_step=50,
            learning_rate=learning_rate,
            network_layers=parsed_layers,
            train_ratio=0.8,
            train_cluster_number=train_cluster_number,
            validation_cluster_number=validation_cluster_number,
        )

    prediction_csv, artifacts = _resolve_prediction_artifacts(
        protein_id=protein_id,
        threshold_dist=threshold_dist,
        trained_model_fold_number=trained_model_fold_number,
        multi_task_learning=multi_task_learning,
        tasks=tasks,
        centrality_feature=centrality_feature,
        cross_validation_fold_number=cross_validation_fold_number,
        epochs=epochs,
        performance_step=50,
        checkpoint_step=50,
        learning_rate=learning_rate,
        network_layers=parsed_layers,
        train_ratio=0.8,
        train_cluster_number=train_cluster_number,
        validation_cluster_number=validation_cluster_number,
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
        f"Rows: {len(df)} | Tasks: {', '.join(tasks)} | Output: {prediction_csv}"
    )
    return summary, df.head(300), prediction_csv, artifacts


def build_app() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="NodeCoder Protein Site Predictor") as app:
        gr.HTML("<div class='hero-title'>NodeCoder Protein Site Predictor</div>")
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
                tasks = gr.CheckboxGroup(
                    choices=AVAILABLE_TASKS,
                    value=["y_Ligand"],
                    label="Prediction Tasks",
                )
                run_btn = gr.Button("Run Prediction", variant="primary")
            with gr.Column(scale=1):
                gr.HTML("<div class='kpi-card'><b>Tips</b><br/>1. Start with one task.<br/>2. Use trained-model fold matching your setup.<br/>3. Outputs are written under results/graph_*A.</div>")

        with gr.Accordion("Advanced Parameters", open=False):
            threshold_dist = gr.Slider(3, 12, value=5, step=1, label="Threshold Distance (Angstrom)")
            trained_model_fold_number = gr.Slider(1, 10, value=1, step=1, label="Trained Model Fold")
            cross_validation_fold_number = gr.Slider(1, 10, value=5, step=1, label="Cross-Validation Folds")
            epochs = gr.Slider(50, 5000, value=2000, step=50, label="Epochs")
            learning_rate = gr.Number(value=0.01, label="Learning Rate")
            network_layers = gr.Textbox(value="38,28,18,8", label="Network Layers (comma-separated)")
            train_cluster_number = gr.Slider(1, 20, value=1, step=1, label="Train Cluster Number")
            validation_cluster_number = gr.Slider(1, 20, value=1, step=1, label="Validation Cluster Number")
            centrality_feature = gr.Checkbox(value=True, label="Use Centrality Feature")

        status = gr.Textbox(label="Run Status")
        preview = gr.Dataframe(label="Prediction Preview (first 300 rows)", wrap=True)
        output_csv = gr.Textbox(label="Prediction CSV Path")
        artifacts = gr.Files(label="Generated Artifact Files")

        run_btn.click(
            fn=run_prediction,
            inputs=[
                protein_id,
                tasks,
                threshold_dist,
                trained_model_fold_number,
                centrality_feature,
                cross_validation_fold_number,
                epochs,
                learning_rate,
                network_layers,
                train_cluster_number,
                validation_cluster_number,
            ],
            outputs=[status, preview, output_csv, artifacts],
        )

    return app


def main() -> None:
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
