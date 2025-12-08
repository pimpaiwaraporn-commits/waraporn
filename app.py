import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================================
# -------- Streamlit UI -------------------
# =========================================
st.title("🌈 2D Electromagnetic Field Simulation (FDTD)")
st.write("ปรับค่าแล้วกด Run Simulation ได้เลย")

# --- PARAMETERS ---
Nx = st.sidebar.slider("Grid Size X", 50, 300, 120)
Ny = st.sidebar.slider("Grid Size Y", 50, 300, 120)
steps = st.sidebar.slider("Simulation Steps", 50, 800, 200)

freq = st.sidebar.s
