
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Electric Field of a +q / -q Dipole")

st.markdown(
    """
    This app shows the electric field of two point charges:
    - **+q** (red) at \((-a, 0)\)  
    - **−q** (blue) at \((+a, 0)\)
    
    You can adjust the separation and magnitude below.
    """
)

# --- Parameters for the dipole ---
a = st.slider("Half separation  a  (distance from origin)", 0.2, 3.0, 1.0, 0.1)
q = st.slider("Negative (-) Charge / Positive (+) Charge  (dimensionless)", 0.5, 3.0, 1.0, 0.1)

# Positions of the charges
x1, y1 = -a, 0.0  # +q
x2, y2 = +a, 0.0  # -q
q1 = +1
q2 = -q

# --- Create a grid for the field ---
x = np.linspace(-4, 4, 41)
y = np.linspace(-4, 4, 41)
X, Y = np.meshgrid(x, y)

def E_point_charge(q, xq, yq, X, Y):
    """
    Electric field of a point charge q at (xq, yq),
    evaluated on grid (X, Y), with k set to 1.
    E = k q r / |r|^3
    """
    dx = X - xq
    dy = Y - yq
    r2 = dx**2 + dy**2
    r = np.sqrt(r2)
    # Avoid division by zero very close to the charge
    r3 = r2 * r
    # Small epsilon to avoid nan at exactly the charge location
    r3 = np.where(r3 == 0, np.nan, r3)

    Ex = q * dx / r3
    Ey = q * dy / r3
    return Ex, Ey

# --- Total field from the two charges ---
Ex1, Ey1 = E_point_charge(q1, x1, y1, X, Y)
Ex2, Ey2 = E_point_charge(q2, x2, y2, X, Y)

Ex = Ex1 + Ex2
Ey = Ey1 + Ey2

# Mask points too close to the charges (for nicer plotting)
mask1 = (X - x1)**2 + (Y - y1)**2 < 0.05
mask2 = (X - x2)**2 + (Y - y2)**2 < 0.05
mask = mask1 | mask2

Ex = np.ma.array(Ex, mask=mask)
Ey = np.ma.array(Ey, mask=mask)

# --- Plot using matplotlib streamplot ---
fig, ax = plt.subplots(figsize=(6, 6))

# Field line coloring by |E|
E_magnitude = np.hypot(Ex, Ey)
color = np.log(E_magnitude)  # log for better contrast

ax.streamplot(
    X, Y, Ex, Ey,
    color=color,
    density=1.2,
    linewidth=1,
    arrowsize=1.2
)

# Plot the charges themselves
ax.scatter([x1], [y1], color="red", s=80, label="+q")
ax.scatter([x2], [y2], color="blue", s=80, label="−q")

ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Electric Field of a +q / −q Dipole (k = 1)")
ax.legend(loc="upper right")

st.pyplot(fig)
