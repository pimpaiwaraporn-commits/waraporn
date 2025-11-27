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
            Ey_total[i, j
