import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. ฟังก์ชันคำนวณสนาม (Field Calculation Functions)
# ----------------------------------------------------------------------

# 1.1 สนามไฟฟ้า (Electric Field E)
def electric_field(x, y, q, x0, y0):
    """คำนวณส่วนประกอบของสนามไฟฟ้า (Ex, Ey) จากประจุ q ที่ (x0, y0)"""
    # E = K * q / r^2. กำหนดให้ K = 1
    
    dx = x - x0
    dy = y - y0
    r_sq = dx**2 + dy**2
    
    # ป้องกันการหารด้วยศูนย์
    r_sq = np.where(r_sq < 1e-12, 1e-12, r_sq)
    r = np.sqrt(r_sq)

    # Ex = q * dx / r^3, Ey = q * dy / r^3
    Ex = q * dx / r**3
    Ey = q * dy / r**3

    return Ex, Ey

# 1.2 สนามแม่เหล็ก (Magnetic Field B)
def magnetic_field(x, y, I, x0, y0):
    """คำนวณส่วนประกอบของสนามแม่เหล็ก (Bx, By) จากกระแส I ที่ (x0, y0)"""
    # B = K' * I / r. กำหนดให้ K' = 1
    
    dx = x - x0
    dy = y - y0
    r_sq = dx**2 + dy**2
    
    # ป้องกันการหารด้วยศูนย์
    r_sq = np.where(r_sq < 1e-12, 1e-12, r_sq)

    # Bx ∝ -dy/r^2, By ∝ dx/r^2 (ตามกฎมือขวา)
    # Bx = I * (-dy / r_sq), By = I * (dx / r_sq)
    Bx = I * (-dy / r_sq)
    By = I * (dx / r_sq)

    return Bx, By

# ----------------------------------------------------------------------
# 2. การตั้งค่ากริด (Grid Setup)
# ----------------------------------------------------------------------
L = 2.0  # ขอบเขตของกราฟ
n = 50   # จำนวนจุด
X, Y = np.meshgrid(np.linspace(-L, L, n), np.linspace(-L, L, n))

# ----------------------------------------------------------------------
# 3. ฟังก์ชันพล็อต (Plotting Function)
# ----------------------------------------------------------------------

def plot_field(field_type, scenario_name, params, L
