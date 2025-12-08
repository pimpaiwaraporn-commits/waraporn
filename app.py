import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------------------------------------------------
# 1. ฟังก์ชันการจำลอง FDTD 2 มิติ
# ----------------------------------------------------------------------

def run_fdtd_simulation(grid_size, time_steps, center_freq, dx):
    """
    ทำการจำลองคลื่นแม่เหล็กไฟฟ้า 2D FDTD ใน Free Space (Ez, Hx, Hy)
    โดยใช้ numpy สำหรับการคำนวณเวกเตอร์ (Vectorized Calculation)
    """
    
    # 1.1 การตั้งค่าพารามิเตอร์ทางฟิสิกส์
    c = 1.0  # ความเร็วแสง (Arb. units)
    
    # คำนวณ dt โดยใช้ค่า dx ที่ผู้ใช้กำหนด (CFL stability condition)
    dt = dx / (c * np.sqrt(2.0))
    
    # ค่าคงที่สำหรับการอัปเดตสนาม: C_H = dt / dx, C_E = dt / dx
    C_H = dt / dx
    C_E = dt / dx

    # 1.2 การตั้งค่ากริดและแหล่งกำเนิด
    Nx = grid_size
    Ny = grid_size
    source_x, source_y = Nx // 2, Ny // 2
    
    # 1.3 การกำหนดตัวแปรสนาม (Field Arrays) ด้วย numpy.zeros
    Ez = np.zeros((Nx, Ny))
    Hx = np.zeros((Nx, Ny - 1))
    Hy = np.zeros((Nx - 1, Ny))

    # 1.4 การเตรียมอาร์เรย์สำหรับเก็บเฟรม
    frame_interval = max(1, time_steps // 50)
    Ez_frames = []

    # 1.5 ฟังก์ชันแหล่งกำเนิด: Ricker Wavelet
    def ricker_wavelet(t, t0, freq):
        tau = np.pi * freq * (t - t0)
        return (1.0 - 2.0 * tau**2) * np.exp(-tau**2)

    t0 = 4.0 / center_freq # จุดศูนย์กลางของ Pulse
    
    # ----------------------------------------------------------------------
    # 2. ลูปเวลา FDTD
    # ----------------------------------------------------------------------
    progress_bar = st.progress(0)
    
    for t in range(time_steps):
        current_time = t * dt
        
        # 2.1 อัปเดตสนามแม่เหล็ก Hx (ใช้ numpy slicing)
        Hx[:] = Hx[:] + C_H * (Ez[:, 1:] - Ez[:, :-1])

        # 2.2 อัปเดตสนามแม่เหล็ก Hy (ใช้ numpy slicing)
        Hy[:] = Hy[:] - C_H * (Ez[1:, :] - Ez[:-1, :])
        
        # 2.3 อัปเดตสนามไฟฟ้า Ez (ใช้ numpy slicing)
        Ez[1:-1, 1:-1] = Ez[1:-1, 1:-1] + C_E * (
            # dHy/dx
            (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) 
            # -dHx/dy
            - (Hx[1:-1, 1:] - Hx[1:-1, :-1]) 
        )
        
        # 2.4 การใส่แหล่งกำเนิด (Source)
        source_value = ricker_wavelet(current_time, t0, center_freq)
        Ez[source_x, source_y] += source_value 
        
        # 2.5 การจัดการขอบเขต (Simple Absorbing Boundary Condition)
        # ป้องกันการสะท้อนจากขอบกริดอย่างง่าย
        Ez[0, :] = Ez[1, :]
        Ez[-1, :] = Ez[-2, :]
        Ez[:, 0] = Ez[:, 1]
        Ez[:, -1] = Ez[:, -2]
        
        # 2.6 การเก็บเฟรมสำหรับการ Animation
        if t % frame_interval == 0 or t == time_steps - 1:
            Ez_frames.append(Ez.copy())
            
        # 2.7 อัปเดตแถบความคืบหน้า
        progress_bar.progress((t + 1) / time_steps)
        
    return Ez.copy(), Ez_frames

# ----------------------------------------------------------------------
# 3. ฟังก์ชันพล็อต (Plotting Function)
# ----------------------------------------------------------------------

def plot_field(field_array, title, ax=None):
    """ฟังก์ชันสำหรับพล็อตสนามแบบ Heatmap ด้วย Matplotlib"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
        
    max_abs_val = np.max(np.abs(field_array))
    if max_abs_val < 1e-10: max_abs_val = 1e-10
        
    im = ax.imshow(field_array.T, cmap='seismic', origin='lower', 
                   vmin=-max_abs_val, 
                   vmax=max_abs_val)
    
    ax.set_title(title)
    ax.set_xlabel('x (grid index)')
    ax.set_ylabel('y (grid index)')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Ez field (arb. units)')
    
    return fig

# ----------------------------------------------------------------------
# 4. อินเทอร์เฟซผู้ใช้ Streamlit (Streamlit UI)
# ----------------------------------------------------------------------
