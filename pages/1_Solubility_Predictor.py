# pages/1_Solubility_Predictor.py (نسخه نهایی)

import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, MolSurf, Draw
import pandas as pd
import joblib

# --- توابع و بارگذاری مدل ---
@st.cache_resource
def load_solubility_model():
    try:
        # --- خط اصلاح شده: بارگذاری مدل نهایی و قدرتمندتر ---
        model = joblib.load('solubility_final_model.joblib')
        return model
    except FileNotFoundError:
        return None

def smiles_to_3d_and_ecfp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol_with_hs = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_with_hs, randomSeed=42, maxAttempts=1000) == -1: return None
        if AllChem.MMFFOptimizeMolecule(mol_with_hs) == -1: return None
        
        asphericity = Descriptors3D.Asphericity(mol_with_hs)
        tpsa = MolSurf.TPSA(mol_with_hs)
        labute_asa = MolSurf.LabuteASA(mol_with_hs)
        ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        
        all_features = np.concatenate(([asphericity, tpsa, labute_asa], ecfp))
        feature_names_3d = ['Asphericity', 'TPSA', 'LabuteASA']
        feature_names_ecfp = [f'bit_{i}' for i in range(1024)]
        all_feature_names = feature_names_3d + feature_names_ecfp
        
        return pd.DataFrame([all_features], columns=all_feature_names)
    except:
        return None
        
def smiles_to_image(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string.strip())
    if mol is None: return None
    return Draw.MolToImage(mol, size=(400, 400))

# --- رابط کاربری صفحه ---
st.set_page_config(page_title="پیش‌بینی محلولیت", page_icon="💧")
st.title("💧 پیش‌بینی محلولیت (LogS)")
st.markdown("پیش‌بینی میزان حلالیت آبی مولکول‌ها با استفاده از مدل **XGBoost**.")
st.divider()

solubility_model = load_solubility_model()

if solubility_model is None:
    st.error("فایل مدل محلولیت (solubility_final_model.joblib) پیدا نشد.")
else:
    smiles_input = st.text_area('ساختار مولکول (SMILES) را وارد کنید:', 'c1ccccc1OC(=O)C')
    
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
            if st.button("پیش‌بینی محلولیت", type="primary", use_container_width=True):
                with st.spinner("در حال پردازش سه‌بعدی و پیش‌بینی..."):
                    input_df = smiles_to_3d_and_ecfp(smiles_input)
                    if input_df is not None:
                        prediction = solubility_model.predict(input_df)
                        st.metric(label="مقدار پیش‌بینی شده برای LogS", value=f"{prediction[0]:.4f}")
                        st.success("آنالیز با موفقیت انجام شد.")
                    else:
                        st.error("خطا در پردازش سه‌بعدی مولکول.")
    else:
        st.info("برای شروع، یک ساختار SMILES وارد کنید.")