import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PHYSICAL AND NUMERICAL PARAMETERS
# =============================================================================
L = 10.0  # Length of the domain (meters)
c = 343.0  # Speed of sound (m/s)
T = 0.02  # Total simulation time (seconds)

Num_Elements = 40  # Number of elements
Num_Nodes = 2 * Num_Elements + 1  # 3 nodes per element (quadratic)
x = np.linspace(0, L, Num_Nodes)  # Node coordinates
dx = L / Num_Elements  # Element length

# Stability condition (CFL) for quadratic elements
dt = 0.5 * (dx / c) / 2.0
time_steps = int(T / dt)

# =============================================================================
# 2. QUADRATIC SHAPE FUNCTIONS (Reference Element: xi in [-1, 1])
# =============================================================================
# N1 = 0.5*xi*(xi - 1),  N2 = 1 - xi^2,  N3 = 0.5*xi*(xi + 1)
# Derivatives with respect to xi:
# dN1 = xi - 0.5,        dN2 = -2*xi,    dN3 = xi + 0.5

# Gauss-Legendre Quadrature (3 points for exact integration of quadratic terms)
gauss_points = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
gauss_weights = np.array([5 / 9, 8 / 9, 5 / 9])


def shape_functions(xi):
    return np.array([0.5 * xi * (xi - 1.0), 1.0 - xi**2, 0.5 * xi * (xi + 1.0)])


def shape_derivatives(xi):
    return np.array([xi - 0.5, -2.0 * xi, xi + 0.5])


# =============================================================================
# 3. GLOBAL MATRIX ASSEMBLY
# =============================================================================
M_global = np.zeros((Num_Nodes, Num_Nodes))
K_global = np.zeros((Num_Nodes, Num_Nodes))

# Jacobian for uniform 1D mesh: dx / d_xi = dx / 2
J = dx / 2.0
invJ = 1.0 / J

for e in range(Num_Elements):
    # Local-to-global node mapping (quadratic elements share boundaries)
    node_indices = [2 * e, 2 * e + 1, 2 * e + 2]

    M_elem = np.zeros((3, 3))
    K_elem = np.zeros((3, 3))

    # Numerical integration over the reference element [-1, 1]
    for gp, qw in zip(gauss_points, gauss_weights):
        N = shape_functions(gp)
        dN_dxi = shape_derivatives(gp)
        dN_dx = dN_dxi * invJ

        M_elem += np.outer(N, N) * J * qw
        K_elem += np.outer(dN_dx, dN_dx) * J * qw


    # Assemble into global matrices
    for i in range(3):
        for j in range(3):
            M_global[node_indices[i], node_indices[j]] += M_elem[i, j]
            K_global[node_indices[i], node_indices[j]] += K_elem[i, j]

# =============================================================================
# 4. INITIAL CONDITIONS & BOUNDARY CONDITIONS
# =============================================================================
# Initial wave profile: Gaussian pulse centered in the middle
u_old = np.exp(-(((x - L / 2) / 0.5) ** 2))
u_curr = np.copy(u_old)  # Assuming zero initial velocity: u_curr = u_old at t=0

# Apply Dirichlet Boundary Conditions (Fixed ends: u(0) = u(L) = 0)
# We zero out the rows/columns or strip them during the solver step.
free_nodes = np.arange(1, Num_Nodes - 1)

# Extract active partitions for internal nodes
M_free = M_global[np.ix_(free_nodes, free_nodes)]
K_free = K_global[np.ix_(free_nodes, free_nodes)]
invM_free = np.linalg.inv(M_free)  # Safe to invert directly for 1D systems

# =============================================================================
# 5. TIME STEPPING LOOP (Central Difference Scheme)
# =============================================================================
u_next = np.zeros(Num_Nodes)

for t in range(time_steps):
    # Extract internal active displacements
    u_curr_free = u_curr[free_nodes]
    u_old_free = u_old[free_nodes]

    # Compute internal force vector: F_int = -c^2 * K * u
    F_int = -(c**2) * np.dot(K_free, u_curr_free)

    # Solve for acceleration: a = M^-1 * F_int
    acceleration = np.dot(invM_free, F_int)

    # Update position: u_next = 2*u_curr - u_old + dt^2 * acceleration
    u_next_free = 2.0 * u_curr_free - u_old_free + (dt**2) * acceleration

    # Insert back into global displacement array (boundaries remain 0)
    u_next[free_nodes] = u_next_free

    # Shift time buffers
    u_old = np.copy(u_curr)
    u_curr = np.copy(u_next)

# =============================================================================
# 6. PLOT VISUALIZATION
# =============================================================================
plt.figure(figsize=(10, 5))
plt.plot(x, np.exp(-(((x - L / 2) / 0.5) ** 2)), "--", label="Initial Pulse (t=0)")
plt.plot(x, u_curr, label=f"FEM Quadratic Solution (t={T}s)", linewidth=2)
plt.title("1D Acoustic Wave Simulation (Quadratic FEM)")
plt.xlabel("Position (m)")
plt.ylabel("Acoustic Pressure / Displacement")
plt.grid(True)
plt.legend()
plt.show(block=True)
