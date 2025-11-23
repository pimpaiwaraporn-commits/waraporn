import numpy as np
import matplotlib.pyplot as plt

# --- กำหนดค่าคงที่และพารามิเตอร์ ---
c = 3e8  # ความเร็วแสง (m/s)
mu0 = 4 * np.pi * 1e-7  # ค่าสภาพซึมผ่านทางแม่เหล็กในสุญญากาศ (H/m)
epsilon0 = 1 / (mu0 * c**2) # ค่าสภาพยอมทางไฟฟ้าในสุญญากาศ (F/m)

E0 = 1.0  # แอมพลิจูดของสนามไฟฟ้า (V/m)
B0 = E0 / c  # แอมพลิจูดของสนามแม่เหล็ก (T)

f = 1e9  # ความถี่ (Hz)
omega = 2 * np.pi * f  # ความถี่เชิงมุม (rad/s)
k = omega / c  # เลขคลื่น (rad/m)

# --- สร้างโดเมนเวลาและพื้นที่ ---
T = 3 / f  # ช่วงเวลาที่จำลอง (ครอบคลุม 3 คาบ)
t = np.linspace(0, T, 500)  # จุดเวลา
x = np.linspace(0, 3 * (2 * np.pi / k), 500) # จุดพื้นที่ (ครอบคลุม 3 ความยาวคลื่น)

# --- คำนวณสนามไฟฟ้า E (ในทิศทาง y) และสนามแม่เหล็ก B (ในทิศทาง z) ---
# E(x, t) = E0 * sin(k*x - omega*t)
# B(x, t) = B0 * sin(k*x - omega*t)

# เลือกเวลา t_snapshot เพื่อดูภาพนิ่ง
t_snapshot = t[50]
E_y = E0 * np.sin(k * x - omega * t_snapshot)
B_z = B0 * np.sin(k * x - omega * t_snapshot)

# --- การแสดงผล (Plotting) ---
plt.figure(figsize=(12, 6))
plt.plot(x, E_y, label=r'Electric Field $E_y$ (V/m)', color='red')
plt.plot(x, B_z * c, label=r'Magnetic Field $\mathbf{B}_z \times c$ (V/m)', color='blue', linestyle='--')
# หมายเหตุ: นำ B คูณ c เพื่อให้มีหน่วยเดียวกับ E และสามารถเปรียบเทียบแอมพลิจูดได้

plt.title(f'1D Electromagnetic Plane Wave at t = {t_snapshot:.2e} s')
plt.xlabel('Position x (m)')
plt.ylabel('Field Strength (V/m)')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.show()
