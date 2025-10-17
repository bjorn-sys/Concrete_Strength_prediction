import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# ⚙️ Streamlit Configuration (MUST BE FIRST STREAMLIT COMMAND)
# -------------------------------------------------------
st.set_page_config(page_title="Concrete Strength Predictor", layout="wide")

# -------------------------------------------------------
# 🧠 Define Model Architecture (same as the trained model)
# -------------------------------------------------------
class DeeperConcreteMLP(nn.Module):
    def __init__(self):
        super(DeeperConcreteMLP, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# -------------------------------------------------------
# 📦 Load Trained Model
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model = DeeperConcreteMLP()
    model.load_state_dict(torch.load("concrete_strength_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# -------------------------------------------------------
# 🎨 App Title and Description
# -------------------------------------------------------
st.title("🧱 Concrete Strength Prediction using PyTorch + ReLU MLP")

st.markdown("""
Predict the **compressive strength (MPa)** of concrete based on its composition.  
Enter material quantities below to get the predicted strength.
""")

# -------------------------------------------------------
# 🧮 User Input Form
# -------------------------------------------------------
with st.form("input_form"):
    st.subheader("Enter Concrete Mix Values")

    col1, col2, col3 = st.columns(3)

    with col1:
        cement = st.number_input("Cement (kg/m³)", 0.0, 600.0, 300.0)
        slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 400.0, 70.0)
        flyash = st.number_input("Fly Ash (kg/m³)", 0.0, 250.0, 50.0)

    with col2:
        water = st.number_input("Water (kg/m³)", 100.0, 250.0, 180.0)
        superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 30.0, 6.0)
        coarseagg = st.number_input("Coarse Aggregate (kg/m³)", 800.0, 1200.0, 970.0)

    with col3:
        fineagg = st.number_input("Fine Aggregate (kg/m³)", 500.0, 1000.0, 770.0)
        age = st.number_input("Age (days)", 1.0, 365.0, 28.0)

    submitted = st.form_submit_button("🔮 Predict Strength")

# -------------------------------------------------------
# 📊 Prediction Logic
# -------------------------------------------------------
if submitted:
    # Create DataFrame from input
    input_data = pd.DataFrame(
        [[cement, slag, flyash, water, superplasticizer, coarseagg, fineagg, age]],
        columns=[
            'Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water',
            'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age'
        ]
    )

    # Load dataset for scaling (ensure you have the same file used for training)
    try:
        sample_dataset = pd.read_csv("concrete_data.csv")
    except FileNotFoundError:
        st.error("❌ 'concrete_data.csv' not found. Please add it to your project folder.")
        st.stop()

    # Normalize input using same scaler logic
    scaler = StandardScaler()
    scaler.fit(sample_dataset.iloc[:, :-1])
    input_scaled = scaler.transform(input_data)

    # Convert to tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Model prediction
    with torch.no_grad():
        prediction = model(input_tensor).item()

    # -------------------------------------------------------
    # 🎯 Display Result
    # -------------------------------------------------------
    st.success(f"**Predicted Concrete Strength:** {prediction:.2f} MPa")
    st.progress(min(prediction / 100, 1.0))
    st.write(f"🧩 Estimated compressive strength: **{prediction:.2f} MPa**")

    st.markdown("---")
    st.caption("Model trained with PyTorch using ReLU activation and Adam optimizer.")
