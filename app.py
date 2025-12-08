# 2D FDTD (TMz) — เสถียรขึ้น (physical units)
import numpy as np
import matplotlib.pyplot as plt

# Grid and physical parameters
nx, ny = 200, 200
dx = dy = 1e-3          # 1 mm grid spacing (meters)
c0 = 299792458.0        # speed of light (m/s)
eps0 = 8.854187817e-12
mu0 = 4*np.pi*1e-7

# Time step (CFL-stable): dt <= 1/(c * sqrt(1/dx^2 + 1/dy^2))
dt = 0.5 * min(dx, dy) / c0

nsteps = 500

# Fields (TMz)
Ez = np.zeros((nx, ny))
Hx = np.zeros((nx, ny))
Hy = np.zeros((nx, ny))

# Uniform free space
eps_r = np.ones((nx, ny))
mu_r = np.ones((nx, ny))

# Update coefficients
cezh = dt / (eps0 * eps_r * dx)   # factor for H->E
chxe = dt / (mu0 * mu_r * dx)     # factor for E->H

# Source: Gaussian-modulated sinusoid
t0 = 60.0
spread = 20.0
freq = 2.0e9       # 2 GHz
omega = 2*np.pi*freq
sx, sy = nx//2, ny//2

# Simple first-order Mur ABC storage
Ez_prev_left = np.zeros(ny)
Ez_prev_right = np.zeros(ny)
Ez_prev_top = np.zeros(nx)
Ez_prev_bottom = np.zeros(nx)

for n in range(nsteps):
    # Hx, Hy update (interior differences)
    Hx[:, :-1] = Hx[:, :-1] - chxe[:, :-1] * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] = Hy[:-1, :] + chxe[:-1, :] * (Ez[1:, :] - Ez[:-1, :])

    # Ez update (interior)
    Ez[1:, 1:] += cezh[1:, 1:] * ((Hy[1:, 1:] - Hy[:-1, 1:]) - (Hx[1:, 1:] - Hx[1:, :-1]))

    # Soft source (additive)
    pulse = np.exp(-0.5 * ((n - t0) / spread) ** 2) * np.sin(omega * n * dt)
    Ez[sx, sy] += pulse

    # Mur ABC (1st order) - edges
    coef = (c0*dt - dx) / (c0*dt + dx)
    Ez[0, :] = Ez_prev_left + coef * (Ez[1, :] - Ez[0, :])
    Ez_prev_left[:] = Ez[1, :].copy()

    Ez[-1, :] = Ez_prev_right + coef * (Ez[-2, :] - Ez[-1, :])
    Ez_prev_right[:] = Ez[-2, :].copy()

    Ez[:, 0] = Ez_prev_top + coef * (Ez[:, 1] - Ez[:, 0])
    Ez_prev_top[:] = Ez[:, 1].copy()

    Ez[:, -1] = Ez_prev_bottom + coef * (Ez[:, -2] - Ez[:, -1])
    Ez_prev_bottom[:] = Ez[:, -2].copy()

# Plot final Ez
plt.figure(figsize=(6,6))
im = plt.imshow(Ez.T, origin='lower', extent=(0,nx,0,ny))
plt.title('Ez field (final step)')
plt.xlabel('x (grid index)')
plt.ylabel('y (grid index)')
plt.colorbar(im, label='Ez (arb. units)')
plt.tight_layout()
plt.show()
