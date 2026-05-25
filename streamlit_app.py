from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_initiator import load_bundle_from_artifact, predict_coords

DEFAULT_ARTIFACT = Path("results/finetune_artifact.pkl")
DEFAULT_FALLBACK_WEIGHTS = Path("DRFold2-model99") / "model_19"

SAMPLE_EXAMPLES = {
    "Short helix": "GGGAAACCC",
    "Purine-rich": "AAGGGAUUGGAA",
    "Pyrimidine-rich": "CCCUUUACCCUU",
}


def sequence_is_valid(seq: str) -> bool:
    allowed = set("ACGUTacgut")
    return bool(seq.strip()) and all(ch in allowed for ch in seq.strip())


def make_arc(p0: np.ndarray, p1: np.ndarray, points: int = 16) -> np.ndarray:
    chord = p1 - p0
    distance = float(np.linalg.norm(chord))
    if distance == 0:
        return np.repeat(p0[None, :], points, axis=0)

    direction = chord / distance
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, up))) > 0.85:
        up = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(direction, up)
    lateral_norm = np.linalg.norm(lateral)
    if lateral_norm == 0:
        lateral = np.array([1.0, 0.0, 0.0])
    else:
        lateral = lateral / lateral_norm

    control = (p0 + p1) / 2.0 + lateral * max(0.5, 0.18 * distance)
    t = np.linspace(0.0, 1.0, points)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t ** 2 * p1


def make_structure_figure(coords: np.ndarray, seq: str) -> go.Figure:
    coords = np.asarray(coords)
    residues = np.arange(1, len(coords) + 1)
    colors = residues.tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="lines+markers+text",
            text=[str(i) for i in residues],
            textposition="top center",
            marker=dict(size=6, color=colors, colorscale="Viridis", showscale=True),
            line=dict(color="#000000", width=5),
            name="Residues",
        )
    )

    for idx in range(len(coords) - 1):
        arc = make_arc(coords[idx], coords[idx + 1])
        fig.add_trace(
            go.Scatter3d(
                x=arc[:, 0],
                y=arc[:, 1],
                z=arc[:, 2],
                mode="lines",
                line=dict(color="rgba(120,120,120,0.45)", width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"Predicted RNA backbone: length {len(seq)}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=720,
    )
    return fig


@st.cache_resource(show_spinner=False)
def load_model(artifact_path: str, device: str):
    path = Path(artifact_path)
    if path.exists():
        return load_bundle_from_artifact(path, device=device)

    from model_initiator import create_model_bundle

    return create_model_bundle(device=device)


def load_sample_sequences() -> dict[str, str]:
    samples = dict(SAMPLE_EXAMPLES)
    seq_path = Path("data") / "train_sequences.csv"
    if seq_path.exists():
        df = pd.read_csv(seq_path, usecols=["target_id", "sequence"]).dropna()
        df = df[df["sequence"].str.len() <= 40].head(5)
        for row in df.itertuples(index=False):
            samples[f"Dataset sample: {row.target_id}"] = row.sequence
    return samples


def main() -> None:
    st.set_page_config(page_title="RNA 3D Predictor", layout="wide")
    st.title("RNA 3D Structure Preview")
    st.write("Enter RNA codes or choose a sample example. The backbone is drawn with residue arcs.")

    samples = load_sample_sequences()
    sample_name = st.selectbox("Sample examples", list(samples.keys()))
    if "rna_sequence" not in st.session_state:
        st.session_state.rna_sequence = samples[sample_name]

    if st.button("Load selected sample"):
        st.session_state.rna_sequence = samples[sample_name]

    seq = st.text_area("RNA sequence", key="rna_sequence", height=120)
    artifact_path = st.sidebar.text_input("Artifact pickle", value=str(DEFAULT_ARTIFACT))
    device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
    atom_index = "C1"
    atom_map = {"C1": 0}

    if not sequence_is_valid(seq):
        st.info("Use only A, C, G, U, or T characters.")
        return

    if st.button("Predict 3D coordinates"):
        with st.spinner("Loading model and predicting..."):
            bundle = load_model(artifact_path, device)
            coords = predict_coords(
                bundle.model,
                seq.strip().upper(),
                bundle.preprocessor,
                bundle.data_module,
                bundle.device,
                atom_index=atom_map[atom_index],
            )

        st.subheader("Predicted coordinates")
        coord_df = pd.DataFrame(coords, columns=["x", "y", "z"])
        coord_df.insert(0, "residue", np.arange(1, len(coord_df) + 1))
        st.dataframe(coord_df, use_container_width=True, height=260)

        fig = make_structure_figure(coords, seq)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()