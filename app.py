import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. ฟังก์ชันคำนวณสนามไฟฟ้า (Electric Field E)
# สูตร: E = K * q / r^2, โดย K = 1/(4*pi*epsilon_0)
# เรากำหนดให้ K = 1 เพื่อความง่ายในการคำนวณและแสดงผล
def electric_field(x, y, q, x0, y0):
    """คำนวณส่วนประกอบของสนามไฟฟ้า (Ex, Ey) ที่จุด (x, y) จากประจุ q ที่ (x0, y0)"""

    dx = x - x0
    dy = y - y0
    r_sq = dx**2 + dy**2

    # ป้องกันการหารด้วยศูนย์ที่ตำแหน่งของประจุ
    # ใช้วิธีแทนที่ค่าที่ r_sq น้อยมากด้วยค่าคงที่เล็กๆ เพื่อหลีกเลี่ยงข้อผิดพลาด
    r_sq = np.where(r_sq < 1e-12, 1e-12, r_sq)

    r = np.sqrt(r_sq)

    # ส่วนประกอบ Ex = q * dx / r^3
    # ส่วนประกอบ Ey = q * dy / r^3
    Ex = q * dx / r**3
    Ey = q * dy / r**3

    return Ex, Ey

# 2. การตั้งค่าและสร้างกริด
L = 2.0  # ขอบเขตของกราฟจาก -L ถึง L
n = 50   # จำนวนจุดในแต่ละแกน (ความละเอียด)
X, Y = np.meshgrid(np.linspace(-L, L, n), np.linspace(-L, L, n))

# 3. ฟังก์ชันหลักสำหรับ Streamlit
def plot_electric_field(charges, title, L):
    """คำนวณและพล็อตสนามไฟฟ้าสำหรับชุดประจุที่กำหนด"""

    # คำนวณสนามรวม
    Ex, Ey = np.zeros_like(X), np.zeros_like(Y)
    for q, x0, y0 in charges:
        Ex_i, Ey_i = electric_field(X, Y, q, x0, y0)
        Ex += Ex_i
        Ey += Ey_i

    # สร้าง Figure ของ Matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))

    # วาดเส้นสนาม (Streamlines)
    # density: ความหนาแน่นของเส้นสนาม
    # linewidth: ความหนาของเส้น
    # arrowsize: ขนาดหัวลูกศร
    ax.streamplot(X, Y, Ex, Ey, density=2, linewidth=1, color='k', arrowsize=1.5)

    # วาดจุดประจุ
    for q, x0, y0 in charges:
        color = 'red' if q > 0 else 'blue'
        label = '+ Charge' if q > 0 else '- Charge'
        ax.plot(x0, y0, 'o', color=color, markersize=10)
        # เพิ่มข้อความเล็กน้อยกำกับชนิดของประจุ
        ax.text(x0, y0 + 0.15, f'{"+" if q > 0 else "-"}{abs(q)}', fontsize=10, ha='center', color=color)

    # การตั้งค่ากราฟ
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)

    return fig

# 4. อินเทอร์เฟซผู้ใช้ Streamlit
st.set_page_config(page_title="Electric Field Visualization", layout="wide")
st.title('⚡ การแสดงภาพสนามไฟฟ้า (Electric Field Visualization)')

st.markdown("""
แอปพลิเคชันนี้ใช้ **Streamlit** และ **Matplotlib** เพื่อแสดงภาพเส้นสนามไฟฟ้าที่เกิดจากชุดของประจุจุด (Point Charges) ในสองมิติ 
เส้นสนามจะชี้ **ออกจากประจุบวก** และชี้ **เข้าสู่ประจุลบ** (สมมติให้ $K=1$)
""")


[Image of Electric Field lines for point charges]


# เลือกตัวอย่างการจัดเรียงประจุ
scenario = st.sidebar.selectbox(
    'เลือกสถานการณ์การจัดเรียงประจุ:',
    ('Single Positive Charge', 'Electric Dipole (+/-)', 'Two Positive Charges (+/+)')
)

# ข้อมูลการจัดเรียงประจุสำหรับแต่ละสถานการณ์
if scenario == 'Single Positive Charge':
    charges = [(1.0, 0.0, 0.0)]
    plot_title = 'สนามไฟฟ้า: ประจุบวกเดี่ยว (+)'
elif scenario == 'Electric Dipole (+/-)':
    charges = [
        (1.0, -0.5, 0.0),  # + Charge
        (-1.0, 0.5, 0.0)   # - Charge
    ]
    plot_title = 'สนามไฟฟ้า: ไดโพลไฟฟ้า (+/-)'
elif scenario == 'Two Positive Charges (+/+)' :
    charges = [
        (1.0, -0.5, 0.0),  # + Charge
        (1.0, 0.5, 0.0)    # + Charge
    ]
    plot_title = 'สนามไฟฟ้า: ประจุบวกคู่ (+/+)'
else:
    charges = []
    plot_title = 'No scenario selected'

# 5. แสดงผลกราฟใน Streamlit
if charges:
    fig = plot_electric_field(charges, plot_title, L)
    st.pyplot(fig)

    st.subheader("รายละเอียดการจัดเรียงประจุ")
    st.dataframe(
        np.array(charges, dtype=[('q', float), ('x', float), ('y', float)]),
        column_order=('q', 'x', 'y'),
        hide_index=True
    )

st.markdown("---")
st.caption("พัฒนาโดยใช้ NumPy และ Matplotlib บนแพลตฟอร์ม Streamlit")
