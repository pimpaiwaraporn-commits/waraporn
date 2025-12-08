# app_fdtd2d_pretty.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import imageio
import os
from tqdm import trange
from time import time

st.set_page_config(layout="wide", page_title="Pretty 2D FDTD Simulator")

# ----------------- Sidebar (controls) -----------------
st.sidebar.title("Simulation controls")

# Grid & time
nx = st.sidebar.slider("Grid X (nx)", 50, 300, 140, step=10)
ny = st.sidebar.slider("Grid Y (ny)", 50, 300, 140, step=10)
dx = dy = st.sidebar.number_input("Cell size dx (m)", value=1e-3, format="%.1e")
c0 = 299792458.0
dt = 0.5 * min(dx, dy) / c0  # CFL-stable (simple)
nsteps = st.sidebar.slider("Time steps", 50, 2000, 300, step=50)

# Source
st.sidebar.markdown("**Source**")
source_type = st.sidebar.selectbox("Type", ["gaussian", "sine"])
src_x = st.sidebar.slider("Source x", 0, nx-1, nx//4)
src_y = st.sidebar.slider("Source y", 0, ny-1, ny//2)
freq = st.sidebar.number_input("Sine freq (Hz)", value=1e9, format="%.0f")
t0 = st.sidebar.slider("Gaussian center t0 (steps)", 1, 500, 60)
spread = st.sidebar.slider("Gaussian spread", 1, 150, 20)

# Objects & materials
st.sidebar.markdown("**Object / Material**")
object_type = st.sidebar.selectbox("Object", ["none", "box", "circle", "ring", "slot", "metal"])
eps_r_obj = st.sidebar.slider("Object εr", 1.0, 20.0, 4.0)
sigma_obj = st.sidebar.number_input("Object σ (S/m)", value=0.0, format="%.2e")

# PML damping
use_pml = st.sidebar.checkbox("Use PML-like absorber", value=True)
pml_frac = st.sidebar.slider("PML thickness (% of min dim)", 1, 15, 6)

# Visualization options
st.sidebar.markdown("**Visualization**")
save_gif = st.sidebar.checkbox("Save GIF animation", value=True)
gif_fps = st.sidebar.slider("GIF fps", 5, 30, 12)
cmap = st.sidebar.selectbox("Colormap", ["seismic", "RdBu", "viridis", "inferno"], index=0)

# Run button
run = st.sidebar.button("Run simulation")

# ----------------- Helper / Setup -----------------
st.title("Pretty 2D FDTD (TMz) — Interactive & Adjustable")
st.write("This demo uses a simple FDTD update for Ez, Hx, Hy (TMz). Adjust parameters in the sidebar then press **Run simulation**.")

def place_object(eps_r, sigma, typ):
    X = np.arange(nx)[:,None]
    Y = np.arange(ny)[None,:]
    if typ == "box":
        x1, x2 = nx//2 - nx//8, nx//2 + nx//8
        y1, y2 = ny//2 - ny//8, ny//2 + ny//8
        eps_r[x1:x2, y1:y2] = eps_r_obj
        sigma[x1:x2, y1:y2] = sigma_obj
    elif typ == "circle":
        cx, cy = nx//2, ny//2
        r = min(nx,ny)//6
        mask = (X-cx)**2 + (Y-cy)**2 <= r*r
        eps_r[mask] = eps_r_obj; sigma[mask] = sigma_obj
    elif typ == "ring":
        cx, cy = nx//2, ny//2
        r1, r2 = min(nx,ny)//8, min(nx,ny)//5
        d2 = (X-cx)**2 + (Y-cy)**2
        mask = (d2 >= r1*r1) & (d2 <= r2*r2)
        eps_r[mask] = eps_r_obj; sigma[mask] = sigma_obj
    elif typ == "slot":
        cx, cy = nx//2, ny//2
        L, W = int(nx*0.5), max(2, int(ny*0.06))
        mask = (np.abs(X-cx) <= L/2) & (np.abs(Y-cy) <= W/2)
        eps_r[mask] = 1.0; sigma[mask] = 0.0
    elif typ == "metal":
        sigma[:,:] = 1e8; eps_r[:,:] = 1.0
    return eps_r, sigma

# ----------------- Run simulation -----------------
if run:
    t_start = time()
    st.info("Running FDTD... (this may take a few seconds depending on grid size and steps)")

    # allocate fields
    Ez = np.zeros((nx, ny), dtype=np.float32)
    Hx = np.zeros_like(Ez)
    Hy = np.zeros_like(Ez)

    eps_r = np.ones((nx, ny), dtype=np.float32)
    sigma = np.zeros((nx, ny), dtype=np.float32)

    eps_r, sigma = place_object(eps_r, sigma, object_type)

    eps0 = 8.854187817e-12
    mu0 = 4*np.pi*1e-7

    # coefficients including simple conductivity
    Ceze = (1 - sigma * dt / (2*eps0*eps_r)) / (1 + sigma * dt / (2*eps0*eps_r))
    Cezh = (dt / (eps0 * eps_r * dx)) / (1 + sigma * dt / (2*eps0*eps_r))
    chxh = 1.0
    chxe = dt / (mu0 * dx)

    # PML-like damping
    damp = np.ones((nx, ny), dtype=np.float32)
    if use_pml:
        pml_th = max(1, int(min(nx,ny) * pml_frac / 100.0))
        sigma_max = 1.0 / (120*np.pi*dx)
        sx = np.zeros(nx); sy = np.zeros(ny)
        for i in range(pml_th):
            val = sigma_max * ((pml_th - i)/pml_th)**2
            sx[i] = val; sx[-1-i] = val
            sy[i] = val; sy[-1-i] = val
        sx3 = sx[:,None]; sy3 = sy[None,:]
        damp = np.exp(- (sx3 + sy3) * dt / (2*eps0))

    # prepare GIF frames
    frames = []
    frame_every = max(1, nsteps // 150)  # limit frames ~150

    probe = (min(nx-1, src_x+10), src_y)  # probe a bit to the right of source
    probe_signal = []

    pbar = st.progress(0)
    status = st.empty()
    for n in range(nsteps):
        # update H
        Hx[:, :-1] -= chxe * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] += chxe * (Ez[1:, :] - Ez[:-1, :])

        # update E
        Ez[1:, 1:] += Cezh[1:,1:] * ((Hy[1:, 1:] - Hy[:-1, 1:]) - (Hx[1:, 1:] - Hx[1:, :-1]))
        # source
        if source_type == "gaussian":
            pulse = np.exp(-0.5 * ((n - t0) / spread) ** 2)
            Ez[src_x, src_y] += pulse
        else:
            Ez[src_x, src_y] += np.sin(2*np.pi*freq*n*dt)

        # apply damping & boundaries
        Ez *= damp

        # record probe
        probe_signal.append(Ez[probe])

        # collect frame
        if save_gif and (n % frame_every == 0):
            # nice normalized image
            im = Ez.copy()
            vmax = max(1e-12, np.percentile(np.abs(im), 99.5))
            img = plt.cm.get_cmap(cmap)((im / (2*vmax)) + 0.5)[:,:,:3]  # RGB
            img = (img * 255).astype(np.uint8)
            frames.append(img)

        # progress
        if n % max(1, nsteps//100) == 0:
            pbar.progress(min(1.0, (n+1)/nsteps))
            status.text(f"Step {n+1}/{nsteps} — max|Ez|={np.max(np.abs(Ez)):.3e}")

    pbar.empty()
    status.empty()

    # produce GIF
    gif_path = os.path.join("fdtd2d_animation.gif")
    if save_gif and frames:
        imageio.mimsave(gif_path, frames, fps=gif_fps)
        st.success(f"Saved animation to {gif_path}")
        st.image(gif_path, use_column_width=True)

    # final beautiful heatmap (Plotly)
    st.subheader("Final Ez field")
    fig = px.imshow(Ez.T, origin='lower', color_continuous_scale=cmap, aspect=nx/ny,
                    labels={'x':'i','y':'j','color':'Ez (arb.)'})
    fig.update_layout(width=600, height=600)
    st.plotly_chart(fig, use_container_width=False)

    # Probe signal + FFT
    st.subheader("Probe signal & spectrum")
    probe_signal = np.array(probe_signal)
    fig2, ax = plt.subplots(2,1,)
