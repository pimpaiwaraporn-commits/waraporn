import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ฟังก์ชันสำหรับคำนวณสนามไฟฟ้าจากจุดประจุ
# ในที่นี้ใช้สูตรสำหรับแรงดึงดูด/ผลักตามกฎของคูลอมบ์ และสมมาตร (bipolar field)
def E_field(x, y, q1, pos1, q2, pos2):
    """
    คำนวณส่วนประกอบของสนามไฟฟ้า E_x และ E_y ที่จุด (x, y)
    จากจุดประจุสองจุด q1 ที่ pos1 และ q2 ที่ pos2.
    """
    # ตำแหน่งประจุ
    x1, y1 = pos1
    x2, y2 = pos2

    # เวกเตอร์ตำแหน่งจากประจุไปยังจุด (x, y)
    r1x, r1y = x - x1, y - y1
    r2x, r2y = x - x2, y - y2

    # ระยะทางยกกำลังสอง (r^2)
    r1_sq = r1x**2 + r1y**2
    r2_sq = r2x**2 + r2y**2

    # หลีกเลี่ยงการหารด้วยศูนย์ใกล้จุดประจุมาก
    # ใช้ค่าคงที่ k=1 เพื่อให้ง่ายต่อการแสดงผล
    k = 1.0
    
    # คำนวณขนาดของสนามไฟฟ้า (k*q/r^2)
    # เราจะคำนวณ E_x และ E_y แยกกัน โดยใช้หน่วยเวกเตอร์ r/r
    
    # สนามจากประจุ 1
    # E_x = k*q*r_x / r^3
    # E_y = k*q*r_y / r^3
    if r1_sq == 0:
        E1x, E1y = 0, 0
    else:
        r1 = np.sqrt(r1_sq)
        E1x = k * q1 * r1x / r1**3
        E1y = k * q1 * r1y / r1**3
    
    # สนามจากประจุ 2
    if r2_sq == 0:
        E2x, E2y = 0, 0
    else:
        r2 = np.sqrt(r2_sq)
        E2x = k * q2 * r2x / r2**3
        E2y = k * q2 * r2y / r2**3

    # สนามไฟฟ้ารวม
    Ex = E1x + E2x
    Ey = E1y + E2y
    
    return Ex, Ey

# --- การตั้งค่า Streamlit ---
st.title("⚡ การจำลองสนามไฟฟ้าจากจุดประจุสองจุด (Bipolar Field)")
st.write("ปรับค่าประจุและตำแหน่งเพื่อดูการเปลี่ยนแปลงของเส้นสนาม")

# แถบด้านข้างสำหรับควบคุม
st.sidebar.header("การตั้งค่าจุดประจุ")

# ประจุ 1 (อาจจะเป็นประจุบวก)
q1 = st.sidebar.slider("ประจุ Q1 (สีแดง)", -10.0, 10.0, 1.0, 0.5)
x1 = st.sidebar.slider("ตำแหน่ง X1", -5.0, 5.0, 1.0, 0.5)
y1 = st.sidebar.slider("ตำแหน่ง Y1", -5.0, 5.0, 0.0, 0.5)
pos1 = (x1, y1)

st.sidebar.markdown("---")

# ประจุ 2 (อาจจะเป็นประจุลบ)
q2 = st.sidebar.slider("ประจุ Q2 (สีน้ำเงิน)", -10.0, 10.0, -1.0, 0.5)
x2 = st.sidebar.slider("ตำแหน่ง X2", -5.0, 5.0, -1.0, 0.5)
y2 = st.sidebar.slider("ตำแหน่ง Y2", -5.0, 5.0, 0.0, 0.5)
pos2 = (x2, y2)

# การตั้งค่าพล็อต
st.sidebar.markdown("---")
st.sidebar.header("การตั้งค่าการแสดงผล")
x_min = st.sidebar.slider("X min", -10.0, 0.0, -5.0)
x_max = st.sidebar.slider("X max", 0.0, 10.0, 5.0)
y_min = st.sidebar.slider("Y min", -10.0, 0.0, -5.0)
y_max = st.sidebar.slider("Y max", 0.0, 10.0, 5.0)
num_points = st.sidebar.slider("ความละเอียด (จำนวนจุดบนแกน)", 10, 50, 20)

# --- การสร้างพล็อต ---
if x_min >= x_max or y_min >= y_max:
    st.error("ค่า Min ต้องน้อยกว่าค่า Max")
else:
    # สร้างเมทริกซ์ของจุด (x, y) สำหรับคำนวณสนาม
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)

    # คำนวณสนามไฟฟ้าที่แต่ละจุด
    # ใช้ np.vectorize เพื่อให้ฟังก์ชัน E_field สามารถรับ Array เป็น Input ได้
    E_field_vec = np.vectorize(E_field)
    Ex, Ey = E_field_vec(X, Y, q1, pos1, q2, pos2)

    # พล็อตด้วย Matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))

    # ใช้ streamplot เพื่อแสดงเส้นสนามไฟฟ้า
    # density=2 หมายถึงมีเส้นสนามหนาแน่นกว่าค่า default
    # color=Ex/np.sqrt(Ex**2 + Ey**2) เป็นการให้สีตามทิศทาง (ไม่จำเป็นแต่เพิ่มความสวยงาม)
    speed = np.sqrt(Ex**2 + Ey**2)
    ax.streamplot(X, Y, Ex, Ey, color=speed, cmap=plt.cm.jet, linewidth=1, density=2, arrowstyle='->', arrowsize=1.5)

    # พล็อตจุดประจุ
    # ประจุบวก (q > 0) เป็นสีแดง, ประจุลบ (q < 0) เป็นสีน้ำเงิน
    ax.plot(x1, y1, 'o', color='red' if q1 >= 0 else 'blue', markersize=10, label=f'Q1 ({q1} C)')
    ax.plot(x2, y2, 'o', color='red' if q2 >= 0 else 'blue', markersize=10, label=f'Q2 ({q2} C)')

    # การตั้งค่าแกน
    ax.set_xlabel("x (dimensions)")
    ax.set_ylabel("y (dimensions)")
    ax.set_title("เส้นสนามไฟฟ้าจากจุดประจุ 2 จุด")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal') # ทำให้แกน x และ y มีมาตราส่วนเท่ากัน
    ax.grid(True, linestyle='--')
    ax.legend()

    # แสดงพล็อตใน Streamlit
    st.pyplot(fig)

    st.caption(f"Q1: {q1} C @ ({x1}, {y1}) | Q2: {q2} C @ ({x2}, {y2})")


