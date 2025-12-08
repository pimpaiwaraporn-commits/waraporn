# app_fdtd_full.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import imageio
import os
from time import time

# ----------------------------
# Utility functions
# ----------------------------
def make_circle_mask(nx, ny, cx, cy, r):
    X = np.arange(nx)[:, None]
    Y = np.arange(ny)[None, :]
    return (X - cx)**2 + (Y - cy)**2 <= r*r

def make_box_mask(nx, ny, x0, y0, x1, y1):
    mask = np.zeros((nx, ny), dtype=bool)
    mask[x0:x1, y0:y1] = True
    return mask

def ricker_wavelet(t, t0, f0):
    """Ricker (Mexican hat) wavelet centered at t0 with central freq f0"""
    tau = (t - t0)
    a = (np.pi * f0 * tau)
    return (1 - 2*a*a) * np.exp(-a*a)

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(layout="wide", page_title="FDTD EM Simulator (full)")
st.title("📡 FDTD Electromagnetic Simulator — Full Feature Demo")

# left column: simulation controls
c1, c2 = st.columns([1,2])
with c1:
    st.header("Simulation control")

    # Mode
    sim_mode = st.selectbox("Mode", ["2D TMz (fast)", "3D (demo, slow)"])

    # Grid and time
    if sim_mode.startswith("2D"):
        nx = st.slider("nx", 40, 400, 160, step=10)
        ny = st.slider("ny", 40, 400, 160, step=10)
        dx = st.number_input("dx (m)", value=1e-3, format="%.1e")
        dy = dx
    else:
        nx = st.slider("nx", 16, 64, 32, step=8)
        ny = nx
        nz = st.slider("nz", 16, 64, 32, step=8)
        dx = st.number_input("dx (m)", value=1e-3, format="%.1e")
        dy = dx
        dz = dx

    steps = st.slider("Time steps", 50, 2000, 400, step=10)

    # Source options
    st.markdown("**Source**")
    source_type = st.selectbox("Source type", ["Gaussian", "Ricker", "CW (sine)"])
    src_freq = st.number_input("Source nominal frequency (Hz)", value=2.4e9, format="%.0f")
    t0 = st.number_input("Pulse center t0 (steps)", value=40)
    spread = st.number_input("Pulse spread", value=12)

    # Objects & materials
    st.markdown("**Objects / Materials**")
    object_preset = st.selectbox("Preset / Add", ["None", "Dielectric block", "Dielectric circle",
                                                  "Ring resonator", "Waveguide (slot)", "Lens", "Metal block"])
    eps_obj = st.number_input("Object εr", value=4.0, format="%.2f")
    sigma_obj = st.number_input("Object conductivity σ (S/m)", value=0.0, format="%.2e")

    # PML
    st.markdown("**Boundary / PML**")
    use_pml = st.checkbox("Use CPML (2D Berenger-like)", value=True)
    pml_thickness = st.slider("PML thickness (cells)", 8, 40, 16)

    # Probes & output
    st.markdown("**Probes & output**")
    probe_mode = st.selectbox("Probe mode", ["single probe", "multiple probes"])
    if probe_mode == "single probe":
        px = st.slider("probe x", 0, nx-1, nx//2)
        py = st.slider("probe y", 0, ny-1, ny//2)
        probes = [(px, py)]
    else:
        # simple default multi-probes along a line
        nprobes = st.slider("n probes", 2, 10, 4)
        probes = []
        base_y = ny//2
        for i in range(nprobes):
            probes.append((int(nx*0.2 + i*nx*0.6/(nprobes-1)), base_y))

    save_gif = st.checkbox("Save GIF animation", value=True)
    downsample_gif = st.slider("GIF downsample (collect every N steps)", 1, 20, 3)

    run_button = st.button("Run Simulation ▶")

# right column: visualization placeholder
with c2:
    st.header("Simulation output")
    viz_placeholder = st.empty()
    probe_placeholder = st.empty()
    fft_placeholder = st.empty()
    info_placeholder = st.empty()

# ----------------------------
# Physics constants / stable dt
# ----------------------------
c0 = 299792458.0
eps0 = 8.854187817e-12
mu0 = 4*np.pi*1e-7

# CFL-stable dt for 2D/3D
if sim_mode.startswith("2D"):
    dt = 0.5 * min(dx, dy) / c0  # conservative
else:
    dt = 0.5 / (c0 * np.sqrt((1/dx**2)+(1/dy**2)+(1/dz**2)))

# ----------------------------
# Setup simulation fields
# ----------------------------
def init_2d_fields(nx, ny):
    Ez = np.zeros((nx, ny), dtype=np.float32)
    Hx = np.zeros((nx, ny), dtype=np.float32)
    Hy = np.zeros((nx, ny), dtype=np.float32)
    eps_r = np.ones((nx, ny), dtype=np.float32)
    sigma_e = np.zeros((nx, ny), dtype=np.float32)
    return Ez, Hx, Hy, eps_r, sigma_e

# CPML helper (2D Berenger-like split-field)
def make_cpml(nx, ny, pml_thickness, dx, dy, dt, sigma_max_scale=1.0):
    # sigma profile (quadratic)
    sigma_x = np.zeros(nx)
    sigma_y = np.zeros(ny)
    sigma_max = sigma_max_scale / (120*np.pi*dx)
    for i in range(pml_thickness):
        x_norm = (pml_thickness - i) / pml_thickness
        sigma_x[i] = sigma_max * x_norm**2
        sigma_x[-1-i] = sigma_max * x_norm**2
        sigma_y[i] = sigma_max * x_norm**2
        sigma_y[-1-i] = sigma_max * x_norm**2

    # create 2D arrays
    sig_x = np.tile(sigma_x[:, None], (1, ny))
    sig_y = np.tile(sigma_y[None, :], (nx, 1))

    # coefficients for split-field update
    # For E-field updates we will use alpha/beta/gamma style factors in update
    kappa = np.ones_like(sig_x)
    b_x = np.exp(- (sig_x / eps0) * dt)
    b_y = np.exp(- (sig_y / eps0) * dt)

    return sig_x, sig_y, b_x, b_y

# ----------------------------
# Main 2D TMz FDTD with CPML and conductive materials
# ----------------------------
def run_fdtd_2d(nx, ny, steps, dx, dy, dt,
                source_type, src_freq, t0, spread,
                eps_r, sigma_e, object_mask,
                probes, use_pml, pml_thickness,
                save_gif, downsample_gif):
    # allocate
    Ez = np.zeros((nx, ny), dtype=np.float32)
    Hx = np.zeros((nx, ny), dtype=np.float32)
    Hy = np.zeros((nx, ny), dtype=np.float32)

    # CPML fields (split fields for Hx/Hy and Ez)
    if use_pml:
        sig_x, sig_y, bx, by = make_cpml(nx, ny, pml_thickness, dx, dy, dt)
        # auxiliary psi fields for split formulation
        psi_Ex_x = np.zeros_like(Ez); psi_Ex_y = np.zeros_like(Ez)
        psi_Ey_x = np.zeros_like(Ez); psi_Ey_y = np.zeros_like(Ez)
        psi_Hx_x = np.zeros_like(Hx); psi_Hx_y = np.zeros_like(Hx)
        psi_Hy_x = np.zeros_like(Hy); psi_Hy_y = np.zeros_like(Hy)
    else:
        sig_x = sig_y = bx = by = None

    # coefficients considering local conductivity sigma_e and eps_r
    aE = (1 - sigma_e * dt / (2 * eps0 * eps_r)) / (1 + sigma_e * dt / (2 * eps0 * eps_r))
    cE = (dt / (eps0 * eps_r)) / (1 + sigma_e * dt / (2 * eps0 * eps_r))
    cH = dt / (mu0 * dx)  # approximate for H updates; use dx for both here (square grid)

    # source location (center-left by default)
    sx = int(nx * 0.25); sy = ny // 2

    # storage for outputs
    frames = []
    probe_records = {p: [] for p in probes}

    # time loop
    for n in range(steps):
        # Update H fields (standard Yee staggered approx)
        # Hx(i,j) -= (dt/mu0) * (Ez(i,j+1) - Ez(i,j)) / dy
        Hx[:, :-1] -= (dt / mu0) * (Ez[:, 1:] - Ez[:, :-1]) / dy
        # Hy(i,j) += (dt/mu0) * (Ez(i+1,j) - Ez(i,j)) / dx
        Hy[:-1, :] += (dt / mu0) * (Ez[1:, :] - Ez[:-1, :]) / dx

        # (optional) add CPML split-field corrections for H if using CPML (skipped detailed split to keep stable)
        # Update Ez from curl(H)
        curlH = (Hy - np.roll(Hy, 1, axis=0)) / dx - (Hx - np.roll(Hx, 1, axis=1)) / dy
        Ez = aE * Ez + cE * curlH

        # inject source
        if source_type == "Gaussian":
            src_val = np.exp(-0.5 * ((n - t0) / spread) ** 2)
        elif source_type == "Ricker":
            src_val = ricker_wavelet(n, t0, src_freq)
        else:  # CW
            src_val = np.sin(2 * np.pi * src_freq * n * dt)
        Ez[sx, sy] += src_val

        # apply object conductivity (already included in aE via sigma_e)
        # apply CPML damping multiplicative factors (approx)
        if use_pml:
            # multiplicative damping using exp(-sigma*dt/eps0) separable approx
            damp = np.exp(- (sig_x + sig_y) * dt / (2 * eps0))
            Ez *= damp

        # record probes
        for p in probes:
            probe_records[p].append(Ez[p])

        # collect frames (downsample for GIF)
        if n % downsample_gif == 0:
            frames.append(Ez.copy())

    return frames, probe_records, Ez

# ----------------------------
# Minimal 3D demo (very small grids)
# ----------------------------
def run_fdtd_3d_demo(nx, ny, nz, steps, dx, dy, dz, dt):
    # Very small 3D Yee-grid demo for illustration only
    Ex = np.zeros((nx, ny, nz), dtype=np.float32)
    Ey = np.zeros_like(Ex)
    Ez = np.zeros_like(Ex)
    Hx = np.zeros_like(Ex)
    Hy = np.zeros_like(Ex)
    Hz = np.zeros_like(Ex)

    # small center source
    sx, sy, sz = nx//2, ny//2, nz//2
    frames = []
    for n in range(steps):
        # simple (and not optimized) update: only Ez from curls of H
        Hx[:-1,:,:] -= (dt/mu0) * (Ez[:-1,1:,:] - Ez[:-1,:-1,:]) / dy
        Hy[:,:-1,:] += (dt/mu0) * (Ez[1:,:-1,:] - Ez[:,:-1,:]) / dx
        # Ez update (rough)
        Ez[1:,1:,1:] += (dt/eps0) * ((Hy[1:,1:,1:] - Hy[:-1,1:,1:]) / dx - (Hx[1:,1:,1:] - Hx[1:,:-1,1:]) / dy)
        # source
        Ez[sx, sy, sz] += np.sin(2*np.pi*1e9*n*dt) * 0.5
        if n % max(1, steps//50) == 0:
            # downsample volume magnitude for visualization
            frames.append(np.sqrt(Ex**2 + Ey**2 + Ez**2).copy())
    return frames, Ez

# ----------------------------
# Run when requested
# ----------------------------
if run_button:
    tstart = time()
    viz_placeholder.info("Running simulation... (this may take some time for large grids)")

    # Initialize material maps and objects
    if sim_mode.startswith("2D"):
        Ez, Hx, Hy, eps_r_map, sigma_e_map = init_2d_fields(nx, ny)

        # place object presets
        object_mask = np.zeros((nx, ny), dtype=bool)
        # Dielectric block
        if object_preset == "Dielectric block":
            x0, x1 = nx//3, 2*nx//3
            y0, y1 = ny//3, 2*ny//3
            object_mask[x0:x1, y0:y1] = True
        elif object_preset == "Dielectric circle":
            object_mask = make_circle_mask(nx, ny, nx//2, ny//2, min(nx,ny)//6)
        elif object_preset == "Ring resonator":
            R1 = min(nx,ny)//8; R2 = R1 + min(nx,ny)//16
            X = np.arange(nx)[:,None]; Y = np.arange(ny)[None,:]
            d2 = (X-nx//2)**2 + (Y-ny//2)**2
            object_mask = (d2 >= R1*R1) & (d2 <= R2*R2)
        elif object_preset == "Waveguide (slot)":
            # vertical slot (conducting walls: make high sigma)
            wg_y0, wg_y1 = ny//3, 2*ny//3
            object_mask[:, : ] = False
            eps_r_map[:, wg_y0:wg_y1] = 1.0
            # create PEC walls at top and bottom of slot by setting sigma very high
            sigma_e_map[:, :wg_y0] = 1e8
            sigma_e_map[:, wg_y1:] = 1e8
        elif object_preset == "Lens":
            # graded index lens (circular)
            X = np.arange(nx)[:,None]; Y = np.arange(ny)[None,:]
            r = np.sqrt((X-nx//2)**2 + (Y-ny//2)**2)
            Rlens = min(nx,ny)//4
            inside = r <= Rlens
            # simple parabolic lens permittivity profile
            eps_r_map[inside] = 1.0 + (eps_obj - 1.0) * (1 - (r[inside]/Rlens)**2)
        elif object_preset == "Metal block":
            object_mask[nx//3:2*nx//3, ny//3:2*ny//3] = True
            sigma_e_map[object_mask] = 1e8
            eps_r_map[object_mask] = 1.0

        # apply dielectric and conductivity where object_mask True (for simple presets)
        eps_r_map[object_mask] = eps_obj
        sigma_e_map[object_mask] = sigma_obj

        # run 2D fdtd
        frames, probe_records, final_Ez = run_fdtd_2d(
            nx, ny, steps, dx, dy, dt,
            source_type, src_freq, t0, spread,
            eps_r_map, sigma_e_map, object_mask,
            probes, use_pml, pml_thickness,
            save_gif, downsample_gif
        )

        elapsed = time() - tstart
        info_placeholder.success(f"Done: {nx}x{ny}, steps={steps}, elapsed={elapsed:.1f}s")

        # Visualization: show last Ez and animation via Plotly slider
        last = frames[-1]
        vmax = np.max(np.abs(last)) if np.max(np.abs(last))>0 else 1e-12

        # Plotly animated heatmap (use fewer frames if too many)
        fig = go.Figure()
        nframes = len(frames)
        max_display = min(nframes, 120)
        step_idx = np.linspace(0, nframes-1, max_display).astype(int)
        for idx, k in enumerate(step_idx):
            visible = (idx==0)
            fig.add_trace(go.Heatmap(z=frames[k].T, zmin=-vmax, zmax=vmax,
                                     colorscale="RdBu", visible=visible, showscale=(idx==0)))
        steps_slider = []
        for i in range(len(step_idx)):
            v = [False]*len(step_idx)
            v[i] = True
            steps_slider.append({"method":"update","args":[{"visible":v}]})
        fig.update_layout(
            title="Ez field (TMz) animation",
            width=700, height=700,
            sliders=[{"steps": steps_slider}],
            updatemenus=[{"type":"buttons","buttons":[{"label":"Play","method":"animate",
                                                     "args":[None, {"frame":{"duration":60,"redraw":True}}]}]}]
        )
        viz_placeholder.plotly_chart(fig, use_container_width=False)

        # Save
