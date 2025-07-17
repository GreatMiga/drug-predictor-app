# pages/2_Toxicity_Predictor.py

import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# --- تعریف معماری و توابع GNN ---
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

def smiles_to_graph(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string.strip())
    if mol is None: return None
    atom_features = [[a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(), a.GetHybridization(), a.GetIsAromatic()] for a in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_indices = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    if not edge_indices: edge_index = torch.empty((2, 0), dtype=torch.long)
    else: edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

def smiles_to_image(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string.strip())
    if mol is None: return None
    return Draw.MolToImage(mol, size=(400, 400))

@st.cache_resource
def load_toxicity_models():
    models = {}
    assay_targets = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    for target in assay_targets:
        try:
            model_path = f'toxicity_gnn_{target}_model.pth'
            model = GCN(num_node_features=5, hidden_channels=128)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            models[target] = model
        except FileNotFoundError:
            pass
    return models

# --- رابط کاربری صفحه ---
st.set_page_config(page_title="پیش‌بینی سمیت", page_icon="☣️")
st.title("☣️ پیش‌بینی سمیت (Tox21)")
st.markdown("پیش‌بینی فعالیت مولکول در ۱۲ مسیر سمیت‌شناسی با استفاده از مدل‌های **GNN**.")
st.divider()

toxicity_models = load_toxicity_models()

if not toxicity_models:
    st.error("هیچ یک از فایل‌های مدل سمیت (.pth) پیدا نشد.")
else:
    toxicity_target = st.selectbox('مسیر سمیت مورد نظر را انتخاب کنید:', list(toxicity_models.keys()))
    smiles_input = st.text_area('ساختار مولکول (SMILES) را وارد کنید:', 'c1ccccc1')

    if smiles_input:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### ساختار دوبعدی:")
            mol_image = smiles_to_image(smiles_input)
            if mol_image:
                st.image(mol_image)
            else:
                st.warning("ساختار SMILES نامعتبر است.")
        
        with col2:
            if st.button("پیش‌بینی سمیت", type="primary", use_container_width=True):
                with st.spinner("در حال پردازش گراف و پیش‌بینی..."):
                    input_graph = smiles_to_graph(smiles_input)
                    if input_graph is not None:
                        model_to_use = toxicity_models.get(toxicity_target)
                        with torch.no_grad():
                            out = model_to_use(input_graph)
                            prediction = out.argmax(dim=1).item()
                            probabilities = F.softmax(out, dim=1).squeeze().tolist()
                        
                        result_text = "سمی (فعال)" if prediction == 1 else "غیرسمی (غیرفعال)"
                        confidence = probabilities[prediction]
                        
                        st.metric(label="نتیجه پیش‌بینی", value=result_text, delta_color="inverse")
                        st.progress(int(confidence * 100))
                        st.write(f"**اطمینان مدل از نتیجه:** {confidence:.2%}")
                    else:
                        st.error("خطا: ساختار SMILES نامعتبر است.")
    else:
        st.info("برای شروع، یک ساختار SMILES وارد کنید.")