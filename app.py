import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from typing import List, Tuple

# ====================
# การตั้งค่าแอป Streamlit (Configuration)
# ====================
st.set_page_page_config(layout="wide", page_title="Electric Field Simulator")
st.title('✨ เครื่องจำลองสนามไฟฟ้าจากประจุจุด 2 มิติ')
st.caption('คำนวณและแสดงผลสนามไฟฟ้าโดยใช้กฎของคูลอมบ์และหลักการซ้อนทับ')

# ค่าคงที่ทางฟิสิกส์
K_COULOMB = 8.9875e9 # ค่าคงที่ของคูลอมบ์ (k)

# ====================
# คลาสและฟังก์ชันคำนวณ (Core Physics Logic)
# ====================

class Charge:
    """Class สำหรับจัดเก็บข้อมูลตำแหน่งและปริมาณประจุ"""
    def __init__(self, x: float, y: float, charge_amount: float):
        self.position = np.array([x, y]) # ตำแหน่ง [x, y]
        self.charge = charge_amount # ปริมาณประจุ (C)

@st.cache_data
def calculate_E_field_single(q: float, r_vec: np.ndarray) -> np.ndarray:
    """คำนวณเวกเตอร์สนามไฟฟ้าจากประจุเดี่ยว E = k * q * r_unit / |r|^2"""
    r_mag = np.linalg.norm(r_vec)
    
    # ป้องกันการหารด้วยศูนย์ (Singularity at the charge location)
    if r_mag < 1e-4: # ใช้ค่าระยะทางขั้นต่ำแทน 0
        return np.array([0.0, 0.0])
        
    r_unit = r_vec / r_mag
    E_mag = K_COULOMB * q / r_mag**2
    return E_mag * r_unit

@st.cache_data
def calculate_total_field(charges_list: List[Charge], x_lim: Tuple[float, float], y_lim: Tuple[float, float], n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """คำนวณสนามไฟฟ้ารวมที่ทุกจุดบนตาราง (Meshgrid)"""
    
    # 1. สร้างตารางจุดสังเกต (Meshgrid)
    x = np.linspace(x_lim[0], x_lim[1], n_points)
    y = np.linspace(y_lim[0], y_lim[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # เตรียม Array สำหรับเก็บส่วนประกอบสนามไฟฟ้ารวม
    Ex_total = np.zeros_like(X, dtype=float)
    Ey_total = np.zeros_like(Y, dtype=float)
    
    num_x = len(x)
    num_y = len(y)

    # 2. วนซ้ำ (Loop) ผ่านทุกจุดสังเกต
    for i in range(num_x):
        for j in range(num_y):
            # ตำแหน่งจุดสังเกตปัจจุบัน
            obs_point = np.array([X[i, j], Y[i, j]])
            E_total = np.array([0.0, 0.0])
            
            # 3. หลักการซ้อนทับ (Superposition)
            for charge in charges_list:
                # เวกเตอร์ระยะทาง r_vec ชี้จากประจุไปยังจุดสังเกต
                r_vec = obs_point - charge.position
                
                # คำนวณสนามไฟฟ้าจากประจุตัวเดียวและบวกสะสม
                E_total += calculate_E_field_single(charge.charge, r_vec)
            
            # 4. เก็บส่วนประกอบของสนามไฟฟ้ารวม
            Ex_total[i, j] = E_total[0]
            Ey_total[i, j] = E_total[1]
            
    # 5. ส่งคืนผลลัพธ์
    return X, Y, Ex_total, Ey_total

# ====================
# การจัดการสถานะ (Session State Management)
# ====================

def initialize_session_state():
    """ตั้งค่าเริ่มต้นของ Session State"""
    if 'charges_data' not in st.session_state:
        # ข้อมูลเริ่มต้น: Electric Dipole
        st.session_state.charges_data = pd.DataFrame([
            {'x (m)': -0.4, 'y (m)': 0.0, 'Charge (C)': 1e-6},
            {'x (m)': 0.4, 'y (m)': 0.0, 'Charge (C)': -1e-6},
        ])

initialize_session_state()

# ====================
# ส่วนควบคุม UI ด้านข้าง (Sidebar Controls)
# ====================

st.sidebar.header('⚙️ การตั้งค่าพื้นที่และการแสดงผล')
span = st.sidebar.slider('ขอบเขตการจำลอง (Span)', 0.5, 3.0, 1.5)
n_points = st.sidebar.slider('ความละเอียดของจุด (N x N)', 15, 40, 25)

st.sidebar.markdown('---')
st.sidebar.header('🔬 การตั้งค่าเวกเตอร์')
col1, col2 = st.sidebar.columns(2)
with col1:
    normalize_vec = st.checkbox('Normalize เวกเตอร์', False)
with col2:
    arrow_color = st.color_picker('สีลูกศร', '#0000FF')

if not normalize_vec:
    scale_factor = st.sidebar.slider('Scale Factor (ความยาวลูกศร)', 1e8, 1e10, 5e9, step=1e8, format='%.1e')
else:
    scale_factor = None
    st.sidebar.info('Normalize ลูกศร: ลูกศรทุกตัวมีความยาวเท่ากัน โดยแสดงเฉพาะทิศทาง')

# ====================
# การจัดการประจุ (Charge Editor)
# ====================

st.subheader('📌 ตารางข้อมูลประจุ (Charge Data)')
st.info('ดับเบิ้ลคลิกที่เซลล์เพื่อแก้ไขค่า X, Y หรือปริมาณประจุ (C) | ใช้ "+" ด้านล่างเพื่อเพิ่มประจุใหม่')

edited_df = st.data_editor(
    st.session_state.charges_data, 
    num_rows="dynamic", 
    key="editor",
    column_config={
        "Charge (C)": st.column_config.NumberColumn(format="%.2e")
    }
)

st.session_state.charges_data = edited_df

# ====================
# การประมวลผลและการแสดงผล (Plotting)
# ====================

current_charges = []
for index, row in st.session_state.charges_data.iterrows():
    try:
        if not np.isnan(row['x (m)']) and not np.isnan(row['y (m)']) and not np.isnan(row['Charge (C)']):
            current_charges.append(Charge(row['x (m)'], row['y (m)'], row['Charge (C)']))
    except Exception as e:
        pass

if not current_charges:
    st.warning("⚠️ กรุณาเพิ่มประจุอย่างน้อยหนึ่งตัวเพื่อเริ่มการจำลอง")
else:
    # 1. คำนวณสนามไฟฟ้ารวม
    x_lim = (-span, span)
    y_lim = (-span, span)

    with st.spinner(f'กำลังคำนวณสนามไฟฟ้าในพื้นที่ {n_points}x{n_points} ({len(current_charges)} ประจุ)...'):
        X, Y, Ex_total, Ey_total = calculate_total_field(current_charges, x_lim, y_lim, n_points)

    # 2. การแสดงผลใน Matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))

    # เตรียมเวกเตอร์สำหรับการแสดงผล
    if normalize_vec:
        Magnitude = np.sqrt(Ex_total**2 + Ey_total**2)
        # Normalize U, V component
        U_plot = np.divide(Ex_total, Magnitude, out=np.zeros_like(Ex_total), where=Magnitude!=0)
        V_plot = np.divide(Ey_total, Magnitude, out=np.zeros_like(V_plot), where=Magnitude!=0)
        final_scale = n_points / 2.0 # Scale เหมาะสมสำหรับ normalized plot
    else:
        U_plot, V_plot = Ex_total, Ey_total
        final_scale = scale_factor

    # วาด Quiver Plot (สนามเวกเตอร์)
    ax.quiver(
        X, Y, U_plot, V_plot, 
        scale=final_scale, 
        color=arrow_color, 
        alpha=0.8, 
        angles='xy', 
        scale_units='xy',
        width=0.003
    )

    # วาดจุดประจุ
    for charge in current_charges:
        color = 'red' if charge.charge > 0 else 'blue'
        
        mag_charge = abs(charge.charge)
        marker_size = max(5, min(20, 10 + np.log10(mag_charge / 1e-7) * 5)) 

        # จุดประจุ
        ax.plot(charge.position[0], charge.position[1], 'o', color=color, markersize=marker_size, markeredgecolor='black', linewidth=1, alpha=0.9)
        
        # ข้อความกำกับ
        charge_text = f"{charge.charge:.2e} C"
        ax.text(charge.position[0] + 0.05, charge.position[1] + 0.05, 
                charge_text, fontsize=9, color=color, weight='bold')

    # ตั้งค่ากราฟ
    ax.set_title(f'Electric Field Map (k = {K_COULOMB:.2e})', fontsize=16)
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)

    # 3. แสดงผลใน Streamlit
    st.pyplot(fig)
