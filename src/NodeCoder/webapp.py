import glob
import gzip
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd
import Bio.PDB

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

# Per-task accent colors: (R, G, B) for the "hot" end of the gradient.
# Base (cold) is always light gray (224, 224, 224).
TASK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "y_Ligand":    (194,  95,   0),   # orange
    "y_ACT_SITE":  (200,  30,  30),   # red
    "y_Nucleic":   ( 30, 100, 200),   # blue
    "y_Peptide":   ( 30, 160,  90),   # green
    "y_Inorganic": (120,  50, 180),   # purple
    "y_Cofactor":  (180, 130,   0),   # gold
    "y_NP_BIND":   ( 20, 150, 180),   # teal
    "y_LIPID":     (180,  80, 140),   # pink
    "y_CARBOHYD":  (100, 160,  30),   # lime
    "y_DISULFID":  (200, 160,   0),   # amber
    "y_MOD_RES":   (140,  80,  40),   # brown
    "y_TRANSMEM":  ( 60, 130, 200),   # sky blue
    "y_VARIANT":   (160,  40, 120),   # magenta
    "y_Artifact":  (120, 120, 120),   # gray
    "y_CHAIN":     ( 80, 180, 160),   # cyan-green
}
_DEFAULT_TASK_COLOR: Tuple[int, int, int] = (194, 95, 0)

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
    """Legacy single-set lookup kept for backward compatibility."""
    selected_key = sorted(selected_tasks)
    catalog = _load_model_catalog()
    for model in catalog["models"]:
        if not model.get("active", True):
            continue
        if sorted(model.get("tasks", [])) == selected_key:
            return model
    available = [", ".join(m.get("tasks", [])) for m in catalog["models"] if m.get("active", True)]
    available_text = " | ".join(available) if available else "none"
    raise ValueError(f"No production model found for selected task set. Available: {available_text}")


def _find_model_for_task(task: str) -> Dict[str, Any]:
    """Return the active single-task model for one task, or raise ValueError."""
    catalog = _load_model_catalog()
    for model in catalog["models"]:
        if not model.get("active", True):
            continue
        if model.get("tasks") == [task]:
            return model
    raise ValueError(f"No active model registered for task '{task}'. Add an entry to model_catalog.json.")

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
.status-box {
    border: 1px solid #d9e2ec;
    border-radius: 12px;
    padding: 10px 12px;
    background: #ffffff;
}
.status-title {
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 6px;
}
.status-success { color: #0f766e; }
.status-warning { color: #b45309; }
.status-error { color: #b91c1c; }
.status-list {
    margin: 0;
    padding-left: 18px;
    color: #52606d;
    font-size: 12px;
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


def _normalize_structure_input(structure_file: Any) -> str:
    if structure_file is None:
        return ""
    if isinstance(structure_file, str):
        return structure_file
    if isinstance(structure_file, dict):
        return structure_file.get("path", "")
    return getattr(structure_file, "name", "")


def _infer_protein_id_from_structure_path(structure_path: str) -> str:
    base = os.path.basename(structure_path)
    for suffix in (".pdb.gz", ".cif.gz", ".ent.gz", ".pdb", ".cif", ".ent"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.strip()


def _generate_structure_visuals(
    protein_id: str,
    tasks: List[str],
    prediction_csv: str,
    structure_file: str,
) -> str:
    if not structure_file:
        return ""
    if not os.path.exists(structure_file):
        return " Warning: provided structure file was not found on disk, so visualization was skipped."

    try:
        from NodeCoder.utilities.visualization import Visualize_Prediction
    except Exception:
        return " Warning: could not import visualization module; structure rendering was skipped."

    try:
        args = SimpleNamespace(
            protein_prediction_fileName=prediction_csv,
            structure_file=structure_file,
            protein_ID=protein_id,
            target_name=tasks,
            path_protein_results=os.path.dirname(prediction_csv) + "/",
        )
        Visualize_Prediction(args)
        return ""
    except Exception:
        return " Warning: structure visualization generation failed; prediction outputs are still available."


def _task_column_prefix(task_name: str) -> str:
    parts = task_name.split("_")
    if len(parts) == 2:
        return parts[1]
    return "_".join(parts[1:])


def _build_status_html(
    headline: str,
    kind: str = "success",
    warnings: List[str] = None,
    skipped: List[str] = None,
) -> str:
    warnings = warnings or []
    skipped = skipped or []

    kind_class = "status-success"
    if kind == "warning":
        kind_class = "status-warning"
    elif kind == "error":
        kind_class = "status-error"

    items = []
    for w in warnings:
        items.append(f"<li><span class='status-warning'>{w}</span></li>")
    if skipped:
        items.append(
            "<li><span class='status-warning'>Skipped tasks (no active model): "
            + ", ".join(skipped)
            + "</span></li>"
        )
    items_html = ""
    if items:
        items_html = "<ul class='status-list'>" + "".join(items) + "</ul>"

    return (
        f"<div class='status-box'>"
        f"<div class='status-title {kind_class}'>{headline}</div>"
        f"{items_html}"
        f"</div>"
    )


def _load_input_task_labels(protein_id: str, task_name: str) -> List[float]:
    """Load per-residue task labels from curated input tasks files only."""
    task_path = os.path.join("./data/input_data/featurized_data/9606/tasks", f"{protein_id}.tasks.csv")
    if os.path.exists(task_path):
        try:
            task_df = pd.read_csv(task_path)
            if task_name in task_df.columns:
                return task_df[task_name].fillna(0).astype(float).tolist()
        except Exception:
            pass

    return []


def _generate_3dmol_visuals(
    protein_id: str,
    tasks: List[str],
    prediction_csv: str,
    structure_file: str,
) -> Tuple[str, str]:
    """Returns (inline_html, warning_str).

    Renders a single interactive 3D viewer with:
    - Task toggle buttons: show/hide each task's coloring
    - Max-wins coloring: residue gets the color of the highest-probability active task
    - Ground Truth toggle: switches coloring to binary GT labels
    - Opaque surface + cartoon both colored
    """
    import io as _io
    import json as _json

    if not structure_file or not os.path.exists(structure_file):
        return "", ""

    try:
        prediction_df = pd.read_csv(prediction_csv)
    except Exception:
        return "", " Warning: prediction CSV could not be read for 3D visualization."

    parser = Bio.PDB.PDBParser(QUIET=True)
    if structure_file.lower().endswith(".gz"):
        with gzip.open(structure_file, "rt") as f:
            raw_structure = parser.get_structure(protein_id, f)
    else:
        raw_structure = parser.get_structure(protein_id, structure_file)

    pdb_model = raw_structure[0]
    chains = sorted(pdb_model.child_list, key=lambda c: len(c), reverse=True)
    if not chains:
        return "", " Warning: no protein chain found for 3D visualization."
    chain = chains[0]
    residues = [r for r in chain.get_residues() if Bio.PDB.is_aa(r, standard=True) and "CA" in r]
    resi_list = [r.get_id()[1] for r in residues]

    # Collect per-task data
    task_resi_probs: Dict[str, Dict[int, float]] = {}  # prefix -> {resi: prob}
    task_resi_gt: Dict[str, Dict[int, float]] = {}     # prefix -> {resi: 0/1}
    task_gt_positive: Dict[str, bool] = {}
    task_colors_map: Dict[str, List[int]] = {}
    task_labels_map: Dict[str, str] = {}
    has_any_gt = False
    warnings: List[str] = []

    for task in tasks:
        col_prefix = _task_column_prefix(task)
        prob_col = f"{col_prefix} Prediction Probability"
        if prob_col not in prediction_df.columns:
            warnings.append(f"missing column '{prob_col}'")
            continue

        probs = prediction_df[prob_col].fillna(0).astype(float).tolist()
        task_resi_probs[col_prefix] = {resi_list[i]: float(probs[i]) for i in range(min(len(resi_list), len(probs)))}

        tc = list(TASK_COLORS.get(task, _DEFAULT_TASK_COLOR))
        task_colors_map[col_prefix] = tc
        task_labels_map[col_prefix] = TASK_LABELS.get(task, col_prefix)

        gt_vals = _load_input_task_labels(protein_id=protein_id, task_name=task)
        if gt_vals:
            task_resi_gt[col_prefix] = {resi_list[i]: float(gt_vals[i]) for i in range(min(len(resi_list), len(gt_vals)))}
            task_gt_positive[col_prefix] = bool(sum(gt_vals) > 0)
            has_any_gt = True

    if not task_resi_probs:
        return "", " Warning: 3D visualization skipped for all tasks (" + "; ".join(warnings) + ")."

    # Write clean PDB (original b-factors don't matter; coloring is done in JS by resi lookup)
    pdb_io = Bio.PDB.PDBIO()
    pdb_io.set_structure(raw_structure)
    pdb_buf = _io.StringIO()
    pdb_io.save(pdb_buf)
    pdb_text = pdb_buf.getvalue().replace("`", "\\`")

    # Serialize data for JS
    task_probs_json = _json.dumps(task_resi_probs)
    task_gt_json = _json.dumps(task_resi_gt)
    task_gt_positive_json = _json.dumps(task_gt_positive)
    task_colors_json = _json.dumps(task_colors_map)
    task_labels_json = _json.dumps(task_labels_map)

    uid = protein_id.lower().replace("-", "_").replace(" ", "_")

    # Build task toggle buttons
    task_btns_html = ""
    for prefix, label in task_labels_map.items():
        tc = task_colors_map[prefix]
        c = f"rgb({tc[0]},{tc[1]},{tc[2]})"
        task_btns_html += (
            f'<button class="nc-task-btn" data-task="{prefix}" '
            f'style="border-color:{c};color:{c}" '
            f'onclick="ncToggleTask_{uid}(this)">{label}</button>'
        )

    gt_btn_html = ""
    if has_any_gt:
        gt_btn_html = (
            f'<button class="nc-view-btn" id="gt-btn-{uid}" '
            f'onclick="ncToggleGT_{uid}(this)">Ground Truth</button>'
        )

    inner_html = f"""<!doctype html>
<html><head><meta charset='utf-8'/>
<script src='https://3Dmol.org/build/3Dmol-min.js'></script>
<style>
body{{margin:0;padding:10px;font-family:'IBM Plex Sans','Segoe UI',sans-serif;background:#f8fafc}}
.nc-task-btn{{padding:4px 12px;font-size:12px;border-radius:16px;border:1.5px solid;background:rgba(255,255,255,0.9);cursor:pointer;opacity:0.35;transition:opacity 0.15s,box-shadow 0.15s;font-weight:600}}
.nc-task-btn.active{{opacity:1;box-shadow:0 0 0 3px rgba(0,0,0,0.08)}}
.nc-view-btn{{padding:4px 12px;font-size:12px;border-radius:16px;border:1.5px solid #334e68;background:transparent;cursor:pointer;color:#334e68;font-weight:600;margin-left:8px}}
.nc-view-btn.active{{background:#334e68;color:#fff}}
</style>
</head><body>
<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:8px">
  <span style="font-size:12px;color:#52606d;white-space:nowrap">Show tasks:</span>
  {task_btns_html}
  {gt_btn_html}
</div>
<div id="gt-hint-{uid}" style="display:none;margin:-2px 0 8px 0;padding:6px 10px;border:1px solid #fed7aa;background:#fffbeb;color:#9a3412;border-radius:8px;font-size:12px"></div>
<div id="viewer-{uid}" style="width:100%;height:520px;position:relative;overflow:hidden;border-radius:12px;border:1px solid #d9e2ec;background:#fff">
  <div id="colorbars-{uid}" style="position:absolute;bottom:10px;left:10px;z-index:10;display:flex;flex-direction:column;gap:5px"></div>
</div>
<script>
(function(){{
  const pdb_{uid} = `{pdb_text}`;
  const taskProbs_{uid} = {task_probs_json};
  const taskGT_{uid} = {task_gt_json};
    const taskGTPos_{uid} = {task_gt_positive_json};
  const taskColors_{uid} = {task_colors_json};
  let activeTasks_{uid} = new Set(Object.keys(taskProbs_{uid}));
  let showGT_{uid} = false;
  let viewer_{uid} = null;
    const gtHintEl_{uid} = document.getElementById('gt-hint-{uid}');
    const colorbarsEl_{uid} = document.getElementById('colorbars-{uid}');
  function colorForResi(resi) {{
    const data = showGT_{uid} ? taskGT_{uid} : taskProbs_{uid};
    let bestTask = null, bestVal = 0;
    activeTasks_{uid}.forEach(function(t) {{
      const val = (data[t] || {{}})[resi];
      if (val !== undefined && val > bestVal) {{ bestVal = val; bestTask = t; }}
    }});
    if (!bestTask || bestVal < 0.001) return 'rgb(220,220,220)';
    const hi = taskColors_{uid}[bestTask];
    const p = Math.min(1, bestVal);
    return 'rgb(' + Math.round(220+(hi[0]-220)*p) + ',' + Math.round(220+(hi[1]-220)*p) + ',' + Math.round(220+(hi[2]-220)*p) + ')';
  }}
  function updatePredBar_{uid}() {{
    if (!colorbarsEl_{uid}) return;
    colorbarsEl_{uid}.innerHTML = '';
    activeTasks_{uid}.forEach(function(t) {{
      const c = taskColors_{uid}[t];
      if (!c) return;
      const rgb = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
      const row = document.createElement('div');
            row.style.cssText = 'display:flex;align-items:center;gap:5px;font-size:10px;color:#243b53;background:rgba(248,250,252,0.92);border:1px solid #d9e2ec;border-radius:6px;padding:3px 6px';
            row.innerHTML = '<span style="font-weight:600">' + t + '</span>'
        + '<span style="display:inline-block;width:80px;height:8px;border-radius:4px;background:linear-gradient(to right, rgb(220,220,220),' + rgb + ');border:1px solid rgba(255,255,255,0.4)"></span>'
                + '<span style="opacity:0.9">0&nbsp;&nbsp;1</span>';
      colorbarsEl_{uid}.appendChild(row);
    }});
  }}
  function recolor_{uid}() {{
    if (!viewer_{uid}) return;
    viewer_{uid}.setStyle({{}}, {{cartoon: {{colorfunc: function(a){{ return colorForResi(a.resi); }} }} }});
    viewer_{uid}.render();
    updatePredBar_{uid}();

        if (!gtHintEl_{uid}) return;
        if (!showGT_{uid}) {{
            gtHintEl_{uid}.style.display = 'none';
            return;
        }}

        if (activeTasks_{uid}.size === 0) {{
            gtHintEl_{uid}.textContent = 'Ground truth mode is ON but no task is selected. Select at least one task.';
            gtHintEl_{uid}.style.display = 'block';
            return;
        }}

        let activeWithGT = [];
        activeTasks_{uid}.forEach(function(t) {{ if ((taskGT_{uid}[t] || null) !== null) activeWithGT.push(t); }});
        if (activeWithGT.length === 0) {{
            gtHintEl_{uid}.textContent = 'Ground truth mode is ON, but no ground-truth labels are available for selected tasks.';
            gtHintEl_{uid}.style.display = 'block';
            return;
        }}

        let anyPositive = false;
        activeWithGT.forEach(function(t) {{ if (taskGTPos_{uid}[t]) anyPositive = true; }});
        if (!anyPositive) {{
            gtHintEl_{uid}.textContent = 'Ground truth labels exist for selected tasks, but all labels are 0 (no annotated positive residues).';
            gtHintEl_{uid}.style.display = 'block';
        }} else {{
            gtHintEl_{uid}.style.display = 'none';
        }}
  }}

  window.ncToggleTask_{uid} = function(btn) {{
    const t = btn.getAttribute('data-task');
    if (activeTasks_{uid}.has(t)) {{ activeTasks_{uid}.delete(t); btn.classList.remove('active'); }}
    else {{ activeTasks_{uid}.add(t); btn.classList.add('active'); }}
    recolor_{uid}();
  }};
  window.ncToggleGT_{uid} = function(btn) {{
    showGT_{uid} = !showGT_{uid};
    btn.classList.toggle('active');
    recolor_{uid}();
  }};

  function init_{uid}() {{
    // Mark all task buttons active on first load to match activeTasks set
    document.querySelectorAll('#colorbars-{uid}');  // no-op, just scoping
    document.querySelectorAll('[data-task]').forEach(function(btn) {{
      const t = btn.getAttribute('data-task');
      if (activeTasks_{uid}.has(t)) btn.classList.add('active');
    }});
    const el = document.getElementById('viewer-{uid}');
    if (!el) return;
    viewer_{uid} = $3Dmol.createViewer(el, {{backgroundColor:'white'}});
    viewer_{uid}.addModel(pdb_{uid}, 'pdb');
    viewer_{uid}.setStyle({{}}, {{cartoon: {{colorfunc: function(a){{ return colorForResi(a.resi); }} }} }});
    viewer_{uid}.addSurface($3Dmol.SurfaceType.MS, {{
      opacity: 0.60,
      colorfunc: function(a){{ return colorForResi(a.resi); }}
    }}, {{}});
    requestAnimationFrame(function() {{
      viewer_{uid}.resize(); viewer_{uid}.zoomTo(); viewer_{uid}.render();
    }});
        recolor_{uid}();
    window.addEventListener('resize', function() {{ viewer_{uid}.resize(); viewer_{uid}.render(); }});
  }}
  if (document.readyState === 'complete') {{ setTimeout(init_{uid}, 50); }}
  else {{ window.addEventListener('load', function() {{ setTimeout(init_{uid}, 50); }}, {{once:true}}); }}
}})();
</script>
</body></html>"""

    srcdoc = inner_html.replace("&", "&amp;").replace('"', "&quot;")
    full_html = f'<iframe srcdoc="{srcdoc}" style="width:100%;height:640px;border:none;border-radius:12px" sandbox="allow-scripts allow-same-origin"></iframe>'
    warn_str = (" Warning: 3D visualization skipped for some tasks (" + "; ".join(warnings) + ").") if warnings else ""
    return full_html, warn_str


def _extract_sequence_from_structure(structure_path: str) -> str:
    import Bio.PDB

    parser = Bio.PDB.PDBParser(QUIET=True)
    if structure_path.lower().endswith(".gz"):
        with gzip.open(structure_path, "rt") as f:
            structure = parser.get_structure("PROT", f)
    else:
        structure = parser.get_structure("PROT", structure_path)

    model = structure[0]
    chains = sorted(model.child_list, key=lambda c: len(c), reverse=True)
    if not chains:
        return ""

    chain = chains[0]
    ppb = Bio.PDB.PPBuilder()
    peptides = ppb.build_peptides(chain)
    sequence = "".join(str(p.get_sequence()) for p in peptides)
    return sequence


def _ensure_test_featurized_inputs(protein_id: str, structure_path: str) -> str:
    from NodeCoder.featurizer import protein_features, protein_tasks

    root = "./data/test_data/featurized_data/9606"
    features_dir = os.path.join(root, "features")
    tasks_dir = os.path.join(root, "tasks")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(tasks_dir, exist_ok=True)

    feature_path = os.path.join(features_dir, f"{protein_id}.features.csv")
    task_path = os.path.join(tasks_dir, f"{protein_id}.tasks.csv")

    if os.path.exists(feature_path) and os.path.exists(task_path):
        return ""

    # If curated input_data files exist for this protein, reuse them for test inference.
    input_feature_path = os.path.join("./data/input_data/featurized_data/9606/features", f"{protein_id}.features.csv")
    input_task_path = os.path.join("./data/input_data/featurized_data/9606/tasks", f"{protein_id}.tasks.csv")
    if os.path.exists(input_feature_path) and os.path.exists(input_task_path):
        shutil.copyfile(input_feature_path, feature_path)
        shutil.copyfile(input_task_path, task_path)
        return " Note: reused curated featurized inputs from data/input_data for this protein."

    if not structure_path or not os.path.exists(structure_path):
        return " Warning: missing structure file; cannot auto-generate inference inputs."

    sequence = _extract_sequence_from_structure(structure_path)
    if not sequence:
        return " Warning: could not extract sequence from structure; preprocessing was skipped."

    pseudo_uniprot = SimpleNamespace(
        entry_name=protein_id,
        sequence=sequence,
        features=[],
        accessions=[protein_id],
    )

    gz_structure_path = structure_path
    temp_gz = None
    if not structure_path.lower().endswith(".gz"):
        temp_gz = tempfile.NamedTemporaryFile(suffix=".pdb.gz", delete=False)
        temp_gz.close()
        with open(structure_path, "rb") as src, gzip.open(temp_gz.name, "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz_structure_path = temp_gz.name

    try:
        feat_df = protein_features.feature_dataframe(pseudo_uniprot, gz_structure_path)
        if feat_df is None:
            return " Warning: feature generation returned no data; preprocessing was skipped."
        feat_df.to_csv(feature_path)

        empty_biolip = pd.DataFrame(columns=["PDB_Chain", "Het", "Uniprot", "Type", "Sequence", "Residues", "Raw"])
        task_df = protein_tasks.task_dataframe(pseudo_uniprot, empty_biolip)
        task_df.to_csv(task_path)
    finally:
        if temp_gz and os.path.exists(temp_gz.name):
            os.remove(temp_gz.name)

    return " Warning: generated missing features/tasks from uploaded structure with unlabeled targets."


def _run_single_task_inference(
    protein_id: str,
    task: str,
    structure_path: str,
) -> Tuple[str, str, List[str]]:
    """Run inference for one task using its registered single-task model.
    Returns (warning_str, prediction_csv_path, artifact_files)."""
    try:
        model = _find_model_for_task(task)
    except ValueError as exc:
        return f" Warning: {exc}", "", []

    try:
        from NodeCoder import predict
    except ImportError:
        return " Error: PyTorch/ML dependencies not installed.", "", []

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

    def _call_predict():
        predict.main(
            protein_ID=protein_id,
            threshold_dist=threshold_dist,
            trained_model_fold_number=trained_model_fold_number,
            multi_task_learning=False,
            Task=[task],
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

    warning = ""
    with _sanitized_argv():
        try:
            _call_predict()
        except FileNotFoundError:
            note = _ensure_test_featurized_inputs(protein_id, structure_path)
            warning += note
            try:
                _call_predict()
            except FileNotFoundError:
                return (
                    f" Error: featurized inputs for '{protein_id}' still missing after preprocessing attempt.",
                    "",
                    [],
                )
            except IndexError:
                warning += " (per-protein metrics skipped)"
        except IndexError:
            warning += " (per-protein metrics skipped)"
        except Exception:
            raise

    prediction_csv, artifacts = _resolve_prediction_artifacts(
        protein_id=protein_id,
        threshold_dist=threshold_dist,
        trained_model_fold_number=trained_model_fold_number,
        multi_task_learning=False,
        tasks=[task],
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
    return warning, prediction_csv, artifacts


def run_prediction(
    protein_id: str,
    tasks: List[str],
    structure_file: Any,
) -> Tuple[str, str, List[str], str]:
    protein_id = (protein_id or "").strip()
    structure_path = _normalize_structure_input(structure_file)
    if not protein_id and structure_path:
        protein_id = _infer_protein_id_from_structure_path(structure_path)

    if not protein_id:
        return _build_status_html(
            "Provide a protein ID, or upload a structure file so the app can infer one from its filename.",
            kind="error",
        ), "", [], ""

    if not tasks:
        return _build_status_html("Please select at least one task.", kind="error"), "", [], ""

    MAX_TASKS = 4
    if len(tasks) > MAX_TASKS:
        return (
            _build_status_html(
                f"Please select at most {MAX_TASKS} tasks at once for reliable prediction quality.",
                kind="warning",
            ),
            "", [], "",
        )

    try:
        from NodeCoder import predict  # noqa: F401 — early import check
    except ImportError:
        return _build_status_html(
            "PyTorch/ML dependencies not installed. Install with: pip install torch torchvision torch-geometric",
            kind="error",
        ), "", [], ""

    # --- Run inference per task using each task's own model ---
    all_warnings: List[str] = []
    all_artifacts: List[str] = []
    task_csvs: Dict[str, str] = {}  # task -> prediction csv path
    succeeded_tasks: List[str] = []
    failed_tasks: List[str] = []

    for task in tasks:
        warning, pred_csv, artifacts = _run_single_task_inference(protein_id, task, structure_path)
        if warning:
            all_warnings.append(f"[{task}] {warning.strip()}")
        if pred_csv and os.path.exists(pred_csv):
            task_csvs[task] = pred_csv
            all_artifacts.extend(artifacts)
            succeeded_tasks.append(task)
        else:
            failed_tasks.append(task)

    if not succeeded_tasks:
        msg = f"All selected tasks failed for '{protein_id}'."
        return _build_status_html(msg, kind="error", warnings=all_warnings), "", [], ""

    # --- Merge prediction CSVs from all successful tasks ---
    first_csv = task_csvs[succeeded_tasks[0]]
    merged_df = pd.read_csv(first_csv)
    for task in succeeded_tasks[1:]:
        other_df = pd.read_csv(task_csvs[task])
        col_prefix = _task_column_prefix(task)
        for col in other_df.columns:
            if col.startswith(col_prefix) and col not in merged_df.columns:
                merged_df[col] = other_df[col].values

    # Save merged CSV next to first task output
    merged_csv = os.path.join(
        os.path.dirname(first_csv),
        f"{protein_id}_merged_{len(succeeded_tasks)}Tasks_results.csv",
    )
    merged_df.to_csv(merged_csv, index=False)
    all_artifacts = sorted(set(all_artifacts))

    # --- 3D visualization with all succeeded tasks ---
    viewer_html, vis_warning = _generate_3dmol_visuals(
        protein_id=protein_id,
        tasks=succeeded_tasks,
        prediction_csv=merged_csv,
        structure_file=structure_path,
    )
    if vis_warning:
        all_warnings.append(vis_warning.strip())
    if not viewer_html:
        all_warnings.append("3D structure visualization requires a structure file to be uploaded.")

    task_model_ids = []
    for task in succeeded_tasks:
        try:
            task_model_ids.append(_find_model_for_task(task)["id"])
        except ValueError:
            pass

    summary = (
        f"Prediction finished for {protein_id}. "
        f"Tasks: {', '.join(succeeded_tasks)} | "
        f"Models: {', '.join(task_model_ids)} | "
        f"Rows: {len(merged_df)} | Output: {merged_csv}"
    )
    kind = "warning" if (all_warnings or failed_tasks) else "success"
    summary_html = _build_status_html(
        summary,
        kind=kind,
        warnings=all_warnings,
        skipped=failed_tasks,
    )
    return summary_html, merged_csv, all_artifacts, viewer_html


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
                    label="Protein ID (optional if structure file is uploaded)",
                    value="KI3L1_HUMAN",
                    placeholder="Example: EGFR_HUMAN",
                )
                structure_file = gr.File(
                    label="Protein Structure File (optional: .pdb or .pdb.gz)",
                    file_count="single",
                    type="filepath",
                )
                task_choices = [(TASK_LABELS[t], t) for t in AVAILABLE_TASKS]
                tasks = gr.CheckboxGroup(
                    choices=task_choices,
                    value=["y_Ligand"],
                    label="Prediction Tasks",
                )
                run_btn = gr.Button("Run Prediction", variant="primary")
            with gr.Column(scale=1):
                gr.HTML("<div class='kpi-card'><b>Tips</b><br/>1. Choose one or more tasks.<br/>2. You can start with Protein ID or upload a structure file and leave Protein ID blank.<br/>3. The app uses the locked production model catalog in src/NodeCoder/model_catalog.json.<br/>4. Outputs are written under results/graph_*A.</div>")

        status = gr.HTML(label="Run Status")
        viewer = gr.HTML(label="3D Structure Viewer", value="<div style='color:#9aa5b4;padding:24px;text-align:center'>Run a prediction with a structure file uploaded to see the interactive 3D viewer.</div>")
        output_csv = gr.Textbox(label="Prediction CSV Path")
        artifacts = gr.Files(label="Generated Artifact Files")

        run_btn.click(
            fn=run_prediction,
            inputs=[
                protein_id,
                tasks,
                structure_file,
            ],
            outputs=[status, output_csv, artifacts, viewer],
        )

    return app


def main() -> None:
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
