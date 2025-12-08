import numpy as np
import matplotlib.pyplot as plt

# 1. ฟังก์ชันคำนวณสนามไฟฟ้า (Electric Field E)
def electric_field(x, y, q, x0, y0):
    """คำนวณส่วนประกอบของสนามไฟฟ้า (Ex, Ey) ที่จุด (x, y) จากประจุ q ที่ (x0, y0)"""

    # K = 1/(4*pi*epsilon_0) กำหนดให้ K = 1 เพื่อความง่ายในการวาด

    dx = x - x0
    dy = y - y0
    r_sq = dx**2 + dy**2

    # ป้องกันการหารด้วยศูนย์ที่ตำแหน่งของประจุ
    r_sq[r_sq < 1e-12] = 1e-12

    r = np.sqrt(r_sq)

    # ขนาดของสนาม E = K * q / r^2
    # ส่วนประกอบ Ex = E * (dx / r) = K * q * dx / r^3
    # ส่วนประกอบ Ey = E * (dy / r) = K * q * dy / r^3

    Ex = q * dx / r**3
    Ey = q * dy / r**3

    return Ex, Ey

# 2. การตั้งค่าและสร้างกริด
L = 2.0  # ขอบเขตของกราฟจาก -L ถึง L
n = 50   # จำนวนจุดในแต่ละแกน (ความละเอียด)
X, Y = np.meshgrid(np.linspace(-L, L, n), np.linspace(-L, L, n))

# --- กราฟที่ 1: สนามของประจุเดี่ยวบวก (Single Positive Charge) ---
plt.figure(figsize=(6, 6))

# ตำแหน่งประจุ: ประจุบวก q1 = 1.0 ที่ (0, 0)
charges1 = [(1.0, 0.0, 0.0)]

# คำนวณสนามรวม
Ex1, Ey1 = np.zeros_like(X), np.zeros_like(Y)
for q, x0, y0 in charges1:
    Ex_i, Ey_i = electric_field(X, Y, q, x0, y0)
    Ex1 += Ex_i
    Ey1 += Ey_i

# วาดเส้นสนามและจุดประจุ
plt.streamplot(X, Y, Ex1, Ey1, density=2, linewidth=1, color='red', arrowsize=1.5)
plt.plot(0.0, 0.0, 'o', color='red', markersize=10, label='+ Charge')

plt.title('Electric Field: Single Positive Charge')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-L, L)
plt.ylim(-L, L)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# --- กราฟที่ 2: สนามของไดโพลไฟฟ้า (Electric Dipole) ---
plt.figure(figsize=(6, 6))

# ตำแหน่งประจุ: q1 = 1.0 ที่ (-0.5, 0), q2 = -1.0 ที่ (0.5, 0)
charges2 = [
    (1.0, -0.5, 0.0),  # + Charge
    (-1.0, 0.5, 0.0)   # - Charge
]

# คำนวณสนามรวม
Ex2, Ey2 = np.zeros_like(X), np.zeros_like(Y)
for q, x0, y0 in charges2:
    Ex_i, Ey_i = electric_field(X, Y, q, x0, y0)
    Ex2 += Ex_i
    Ey2 += Ey_i

# วาดเส้นสนามและจุดประจุ
plt.streamplot(X, Y, Ex2, Ey2, density=2, linewidth=1, color='k', arrowsize=1.5)
plt.plot(-0.5, 0.0, 'o', color='red', markersize=10, label='+ Charge')
plt.plot(0.5, 0.0, 'o', color='blue', markersize=10, label='- Charge')

plt.title('Electric Field: Electric Dipole (+/-)')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-L, L)
plt.ylim(-L, L)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# --- กราฟที่ 3: สนามของประจุบวกคู่ (Two Positive Charges) ---
plt.figure(figsize=(6, 6))

# ตำแหน่งประจุ: q1 = 1.0 ที่ (-0.5, 0), q2 = 1.0 ที่ (0.5, 0)
charges3 = [
    (1.0, -0.5, 0.0),  # + Charge
    (1.0, 0.5, 0.0)    # + Charge
]

# คำนวณสนามรวม
Ex3, Ey3 = np.zeros_like(X), np.zeros_like(Y)
for q, x0, y0 in charges3:
    Ex_i, Ey_i = electric_field(X, Y, q, x0, y0)
    Ex3 += Ex_i
    Ey3 += Ey_i

# วาดเส้นสนามและจุดประจุ
plt.streamplot(X, Y, Ex3, Ey3, density=2, linewidth=1, color='red', arrowsize=1.5)
plt.plot(-0.5, 0.0, 'o', color='red', markersize=10, label='+ Charge')
plt.plot(0.5, 0.0, 'o', color='red', markersize=10, label='+ Charge')

plt.title('Electric Field: Two Positive Charges (+/+)')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-L, L)
plt.ylim(-L, L)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
