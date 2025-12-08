import streamlit as st
import numpy as np
import plotly.graph_objects as go
import imageio
import os

# ==================================================
# Sidebar UI
# ==================================================
st.title("🌈 2D Electromagnetic Field Simulation (FDTD)")

Nx = st.sidebar.slider("Grid size X", 50, 300, 200)
Ny = st.sidebar.slider("Grid size Y", 50, 300, 200)
steps = st.sidebar.slider("Timesteps", 50, 600, 300)

freq = st.sidebar.slider("Source frequency (Hz)", 1e8, 5e9, 2e9)
eps_obj = st.sidebar.slider("Object permittivity εr", 1.0, 12.0, 6.0)

object_type = st.sidebar.selectbox("Object Shape", ["none", "rect", "circle"])

source_x = st.sidebar.slider("Source X position", 0, Nx - 1, Nx // 2)
source_y = st.sidebar.slider("Source Y position", 0, Ny - 1, Ny // 2)

run_sim = st.sidebar.button("🚀 Run Simulation")

# ==================================================
# FDTD Simulation function
# ==================================================
def run_fdtd(Nx, Ny, steps, freq, eps_obj, object_type, source_x, source_y):

    # constants
    c = 3e8
    dx = dy = 1e-3
    dt = 1 / (c * np.sqrt(2)) * 0.8

    Ez = np.zeros((Nx, Ny))
    Hx = np.zeros((Nx, Ny))
    Hy = np.zeros((Nx, Ny))

    eps = np.ones((Nx, Ny))

    # ---------------------------
    #   Insert object
    # ---------------------------
    if object_type == "rect":
        eps[Nx // 3:2 * Nx // 3, Ny // 3:2 * Ny // 3] = eps_obj

    elif object_type == "circle":
        cx, cy = Nx // 2, Ny // 2
        R = Nx // 5
        for i in range(Nx):
            for j in range(Ny):
                if (i - cx)**2 + (j - cy)**2 < R**2:
                    eps[i, j] = eps_obj

    frames = []

    # ---------------------------
    #     FDTD Loop
    # ---------------------------
    for n in range(steps):

        # Update H fields
        Hx[:, :-1] -= (dt / dy) * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] += (dt / dx) * (Ez[1:, :] - Ez[:-1, :])

        # Curl of H
        curlH = (Hy - np.roll(Hy, 1, axis=0)) - (Hx - np.roll(Hx, 1, axis=1))

        # Update Ez
        Ez += (dt / eps) * curlH

        # Source
        Ez[source_x, source_y] += np.sin(2 * np.pi * freq * n * dt) * 2.0

        if n % 3 == 0:
            frames.append(Ez.copy())

    return frames

# ==================================================
# Run Simulation
# ==================================================
if run_sim:

    st.info("⏳ Running simulation... please wait")

    frames = run_fdtd(Nx, Ny, steps, freq, eps_obj, object_type, source_x, source_y)

    st.success("Simulation completed!")

    # ==================================================
    # Plotly interactive animation
    # ==================================================
    fig = go.Figure()

    for i, f in enumerate(frames):
        fig.add_trace(go.Heatmap(
            z=f,
            zmin=-1,
            zmax=1,
            colorscale="Turbo",
            visible=(i == 0)
        ))

    # slider frames
    steps_slider = []
    for i in range(len(frames)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(frames)}]
        )
        step["args"][0]["visible"][i] = True
        steps_slider.append(step)

    fig.update_layout(
        title="Ez field (FDTD 2D)",
        width=650,
        height=650,
        sliders=[{"steps": steps_slider}],
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 40, "redraw": True}}])
            ]
        }]
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==================================================
    # Save GIF
    # ==================================================
    gif_path = "fdtd.gif"
    images = []

    for f in frames:
        f_norm = (255 * (f - f.min()) / (f.max() - f.min())).astype(np.uint8)
        images.append(f_norm)

    imageio.mimsave(gif_path, images, fps=20)
    st.success("GIF generated successfully!")

    with open(gif_path, "rb") as file:
        st.download_button("⬇ Download GIF", file, "fdtd.gif", "image/gif")

else:
    st.info("👈 กดปุ่ม **Run Simulation** เพื่อเริ่มการจำลอง")
