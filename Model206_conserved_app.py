import streamlit as st
import pandas as pd
import joblib
import re

st.set_page_config(page_title="Mir v1 — Scenario Model", page_icon="🧬")
st.title("🧬 miRNA Upregulation Predictor — v1 Scenario")
st.caption("Random Forest · OneHotEncoder · No seed family · Synthetic data")

# ── Load model
@st.cache_resource
def load_model():
    return joblib.load('Mir_v1_scenario_model.pkl')

model_pipeline = load_model()

# ── History
if 'history' not in st.session_state:
    st.session_state.history = []

# ── miRNA group resolver (no lookup — derive from name)
def resolve_group(mirna_name: str) -> str:
    """Strip species prefix and arm suffix to get simplified group."""
    name = mirna_name.strip()
    name = re.sub(r'^[a-z]{3}-', '', name.lower())   # strip hsa-, mmu-
    name = re.sub(r'-(5p|3p)$', '', name)             # strip -5p/-3p
    name = re.sub(r'-[123]$', '', name)               # strip locus -1/-2/-3
    return name

# ── Inputs
st.subheader("Enter Prediction Inputs")
mirna_input = st.text_input("miRNA name or accession number",
                             placeholder="e.g. hsa-miR-21, miR-155, MIMAT0000076")
parasite    = st.selectbox("Parasite",  options['parasite'])
organism    = st.selectbox("Organism",  options['organism'])
cell_type   = st.selectbox("Cell type", options['cell_type'])

time = st.number_input(
    "Time (hours post-infection)",
    min_value=int(min(options['time'])),
    max_value=int(max(options['time'])),
    value=int(options['time'][0]),
    step=1
)
# ── Show resolved group
if mirna_input:
    group = resolve_group(mirna_input)
    st.info(f"**Resolved miRNA group:** `{group}`")

# ── Predict
if st.button("Predict", type="primary"):
    if not mirna_input.strip():
        st.warning("Please enter a miRNA name.")
    else:
        group    = resolve_group(mirna_input)
        scenario = f"{parasite.strip()}_{cell_type.strip()}"

        input_df = pd.DataFrame([{
            'microrna_group_simplified': group,
            'parasite':                 parasite,
            'organism':                 organism,
            'cell type':                cell_type,
            'time':                     time,
            'scenario':                 scenario,
        }])

        proba = model_pipeline.predict_proba(input_df)[0][1]
        pred  = int(proba >= 0.5)
        label = "⬆️ Upregulated" if pred == 1 else "⬇️ Downregulated"
        color = "green" if pred == 1 else "red"

        st.markdown(f"### Prediction: :{color}[{label}]")
        st.metric("Confidence", f"{proba*100:.1f}%")

        # Add to history
        st.session_state.history.append({
            "miRNA":      mirna_input.strip(),
            "Group":      group,
            "Parasite":   parasite,
            "Organism":   organism,
            "Cell type":  cell_type,
            "Time (h)":   time,
            "Prediction": label,
            "Confidence": f"{proba*100:.1f}%",
        })

# ── History table
if st.session_state.history:
    st.subheader("Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    if st.button("Clear history"):
        st.session_state.history = []
