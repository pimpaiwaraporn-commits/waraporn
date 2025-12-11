import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --- 1. ฟังก์ชันคำนวณสนามไฟฟ้าใน 3 มิติ ---
@st.cache_data
def calculate_E_field_3d(X, Y, Z, q, q_pos):
    """
    คำนวณเวกเตอร์สนามไฟฟ้า (Ex, Ey, Ez) จากประจุจุดเดียวใน 3 มิติ
    ใช้ @st.cache_data เพื่อป้องกันการคำนวณซ้ำเมื่อพารามิเตอร์ไม่เปลี่ยน
    """
    x0, y0, z0 = q_pos

    dx = X - x0
    dy = Y - y0
    dz = Z - z0

    r_squared = dx**2 + dy**2 + dz**2

    # ป้องกันการหารด้วยศูนย์
    r_squared = np.where(r_squared < 1e-12, 1e-12, r_squared)
    r = np.sqrt(r_squared)

    # ขนาดของสนามไฟฟ้า E = k * q / r^2 (k = 1 เพื่อความง่าย)
    E_magnitude = q / r_squared

    # ส่วนประกอบของสนามไฟฟ้า
    Ex = E_magnitude * (dx / r)
    Ey = E_magnitude * (dy / r)
    Ez = E_magnitude * (dz / r)

    return Ex, Ey, Ez

# --- 2. Streamlit UI และการตั้งค่า ---

st.set_page_config(page_title="3D Electric Field Visualizer (Plotly)", layout="wide")
st.title("🔌 3D Electric Field Visualization")
st.caption("การแสดงผลเวกเตอร์สนามไฟฟ้า 3 มิติรอบประจุจุดเดียวโดยใช้ Plotly")

# --- 3. Sidebar สำหรับการตั้งค่าพารามิเตอร์ ---

with st.sidebar:
    st.header("Charge Parameters")
    charge_q = st.slider("Charge Value (q)", -10.0, 10.0, 5.0, 0.5)
    
    st.subheader("Charge Position (x0, y0, z0)")
    lim_pos = 2.0
    charge_x0 = st.slider("x0", -lim_pos, lim_pos, 0.0, 0.1)
    charge_y0 = st.slider("y0", -lim_pos, lim_pos, 0.0, 0.1)
    charge_z0 = st.slider("z0", -lim_pos, lim_pos, 0.0, 0.1)
    
    charge_pos = (charge_x0, charge_y0, charge_z0)

    st.header("Grid Settings")
    n_points = st.slider("Resolution (Points per axis)", 5, 20, 10, 1)
    lim = st.slider("Boundary (-L to L)", 1.0, 5.0, 3.0, 0.5)
    
    cone_size_ref = st.slider("Vector Size Multiplier", 0.1, 2.0, 0.5, 0.1)
    

# --- 4. การคำนวณและสร้างกราฟ ---

if st.button("Generate 3D Field") or True: # True: เพื่อให้รันอัตโนมัติเมื่อเปิดแอป

    # 4.1 สร้างกริด
    x_range = np.linspace(-lim, lim, n_points)
    y_range = np.linspace(-lim, lim, n_points)
    z_range = np.linspace(-lim, lim, n_points)
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')

    # 4.2 คำนวณสนามไฟฟ้า
    Ex, Ey, Ez = calculate_E_field_3d(X, Y, Z, charge_q, charge_pos)

    # 4.3 สร้าง Trace สำหรับประจุ
    charge_trace = go.Scatter3d(
        x=[charge_pos[0]], y=[charge_pos[1]], z=[charge_pos[2]],
        mode='markers',
        marker=dict(
            size=15,
            color='red' if charge_q >= 0 else 'blue',
            symbol='circle',
            opacity=1.0
        ),
        name=f'Point Charge q = {charge_q:.1f}'
    )

    # 4.4 สร้าง Cone trace สำหรับสนามไฟฟ้า
    # คำนวณขนาด (Magnitude) เพื่อใช้ใน colorscale
    E_mag_flat = np.sqrt(Ex.flatten()**2 + Ey.flatten()**2 + Ez.flatten()**2)
    
    # กำหนดค่าที่มากที่สุดของสี
    max_e_mag = np.max(E_mag_flat)
    
    field_trace = go.Cone(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        u=Ex.flatten(), v=Ey.flatten(), w=Ez.flatten(),
        sizemode="absolute",
        sizeref=cone_size_ref, 
        anchor="tip",
        colorscale='Hot',
        cmin=0,
        cmax=max_e_mag * 0.5, # ปรับ cmax เพื่อให้เห็นความแตกต่างของสีชัดเจนขึ้น
        showscale=True,
        colorbar=dict(title='|E| Magnitude'),
        name='Electric Field Vector'
    )

    # 4.5 จัดองค์ประกอบของกราฟ
    fig = go.Figure(data=[charge_trace, field_trace])

    fig.update_layout(
        title=f'3D Electric Field (q = {charge_q:.1f})',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            # ปรับมุมมองและขอบเขต
            aspectmode='cube',
            xaxis=dict(range=[-lim, lim]),
            yaxis=dict(range=[-lim, lim]),
            zaxis=dict(range=[-lim, lim])
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # 4.6 แสดงผลใน Streamlit
    st.plotly_chart(fig, use_container_width=True)
