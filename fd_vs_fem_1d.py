# %%
#
# Daniel Rodrigues Pipa
# Universidade Tecnológica Federal do Paraná
# 2026
#

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.signal as ss
from tqdm import tqdm
import sympy as sym
import chaospy
from numpy.polynomial import legendre
from findiff import coefficients

# FEM polynomial order of shape function
polyOrd = 5
SAVE_MOV = True
# SAVE_MOV = False

cs = [6000, 1500]  # Propagation speeds
dxs = [polyOrd * 100e-6, polyOrd * 30e-6]  # Spatial step
dt = 3e-9  # Time step
Lx = 30e-3  # Spatial length
Lt = 10e-6  # Temporal length
f0 = 5e6  # Central frenquency

# Grid points
xgrid = np.arange(0, Lx/2, dxs[0])
xgrid = np.append(xgrid, np.arange(xgrid[-1] + dxs[0], Lx + dxs[1], dxs[1]))

Mx = len(xgrid)  # Number of grid points
Me = Mx - 1  # Number of elements
Mg = Me * polyOrd + 1  # Number of collocation points (global matrices)


def gll_nodes_and_weights(n: int):
    """
    Returns the n Gauss-Lobatto-Legendre nodes and weights over [-1, 1].
    """
    if n < 2:
        raise ValueError("Number of nodes n must be >= 2.")

    # Handle the minimal case manually
    if n == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    # 1. Initialize the coefficients for P_{n-1}
    # Polynomial sequence of length n represents degree n-1
    coeffs = np.zeros(n)
    coeffs[-1] = 1.0

    # 2. Get the derivative of P_{n-1} and find its internal roots
    P_nm1 = legendre.Legendre(coeffs)
    internal_nodes = P_nm1.deriv().roots()

    # 3. Combine endpoints [-1, 1] with the internal roots
    nodes = np.hstack(([-1.0], internal_nodes, [1.0]))

    # 4. Calculate the corresponding GLL weights
    # Formula: w_i = 2 / (n * (n - 1) * [P_{n-1}(x_i)]^2)
    P_vals = P_nm1(nodes)
    weights = 2.0 / (n * (n - 1) * P_vals**2)

    return nodes, weights

nodes, _ = gll_nodes_and_weights(polyOrd + 1)  # Internal nodes

def build_xfem():
    """Build all collocation points (grid points with internal nodes)"""
    xfem = np.zeros(Mg)
    nodes_ = ((nodes+1)/2)
    for i in range(Mx-1):
        h = xgrid[i+1] - xgrid[i]
        for p in range(polyOrd):
            xfem[polyOrd*i+p] = xgrid[i] + h * nodes_[p]
    xfem[-1] = xfem[-2] + h * nodes_[1]
    return xfem

xfem = build_xfem()

C = max(cs) * dt / min(dxs)
print(f"Courant number: {C}")
if C > 1:
    raise ValueError(f"Courant number {C} > 1")

# Place the propagation speeds
c2fem = np.zeros(Mg)
c2fem[xfem <= Lx/2] = cs[0]**2
c2fem[xfem > Lx/2] = cs[1]**2

# For comparison purposes, set up a Finite Differences Method
# with the same number of points of collocations points
dx = Lx / (Mg - 1)
xfdmm = np.arange(Mg) * dx
c2fdmm = np.zeros(Mg)
c2fdmm[xfdmm <= Lx/2] = cs[0]**2 * dt**2 / dx**2
c2fdmm[xfdmm > Lx/2] = cs[1]**2 * dt**2 / dx**2

Mt = round(Lt / dt)  # Number of temporal points
t = np.arange(Mt) * dt  # Temporal points

# Source term
t0 = 1.5 / f0
# s = ss.gausspulse(t - t0, f0, bw)
def ricker(t: npt.NDArray, f0: float):
    sigma = .25 / f0
    tmp = (1 - (t/sigma)**2) * np.exp(-t**2 / (2 * sigma**2))
    return tmp / np.max(tmp)
s = ricker(t - t0, f0)
# plt.plot(t, s)
# plt.show(block=True)


# Build shape functions symbolically
# TODO: Changing this to quadrature methods will yield to Spectral Elements Methods!

xsym, hsym = sym.symbols("xsym, hsym")
xisym = (2 * xsym / hsym)

def buildShapeFunctions(p: int):
    """Lagrange polynomials to be used as shape functions"""
    Ne = sym.Matrix(p+1, 1, np.ones(p+1))
    xi = nodes #np.linspace(-1, 1, p+1)
    for i in range(p+1):
        for j in range(p+1):
            if i != j:
                Ne[i, 0] *= (xisym - xi[j]) / (xi[i] - xi[j])
    return Ne

Ne = buildShapeFunctions(polyOrd)

# # Plot shape functions
# h = 1
# x_ = np.arange(-h/2, h/2, .01)
# for ne in Ne:
#     plt.plot(x_, sym.lambdify((xsym, hsym), ne, "numpy")(x_, h))
# plt.grid(True)
# plt.show(block=True)

def buildMat(Ne: sym.Matrix):
    """Generic function to build the mass and the stiffness matrix"""
    M = np.zeros((Mg, Mg))
    Me_lmbd = sym.lambdify((hsym), (Ne @ Ne.T).integrate((xsym, -hsym/2, hsym/2)), "numpy")
    for i in tqdm(range(Me)):
        x_ip1 = xgrid[i + 1] if i < Me else xgrid[-1]-1
        x_i = xgrid[i]
        r = np.arange(polyOrd * i, polyOrd * i + polyOrd + 1)
        M[np.ix_(r, r)] += Me_lmbd(x_ip1 - x_i)
    return M


def buildf(xs: float):
    """Build source vector"""
    f_i = sym.Piecewise((1, sym.And(-hsym <= xsym, xsym <= hsym)), (0, True))
    fe_lmbd = sym.lambdify((hsym), (Ne * f_i).integrate((xsym, -hsym / 2, hsym / 2)), "numpy")
    f = np.zeros(Mg)
    k = np.argmin(np.abs(xs - xfem))
    f[k] = np.sum(fe_lmbd(xfem[k + polyOrd] - xfem[k]))
    return f

# def build_plot_weights():
#     x_ = np.arange(-.5, .5, 1/polyOrd)
#     N = len(x_)
#     plot_weights = np.zeros(N)
#     for i in range(N):
#         plot_weights[i] = sym.lambdify((xsym, hsym), Ne[i], "numpy")(x_[i], 1)
#     return plot_weights
#
# print(build_plot_weights())
#%%

dNe = Ne.diff(xsym)

print("Building matrices...")
M = buildMat(Ne)  # Mass matrix
K = buildMat(dNe)  # Stiffness matrix
f = buildf(Lx/3)  # Source vector
Minv = np.linalg.inv(M)
print("Done building matrices.")

# Debug
# DEBUG_SHOW = True
DEBUG_SHOW = False

if DEBUG_SHOW:
    plt.figure()
    plt.imshow(K)
    plt.title("K")
    plt.colorbar()
    
    plt.figure()
    plt.imshow(M)
    plt.title("M")
    plt.colorbar()
    
    plt.figure()
    plt.imshow(Minv)
    plt.title("Minv")
    plt.colorbar()

    plt.show(block=True)


#%%

# Explicit time marching simulations
# fdmacc = polyOrd + 1
# fdmacc += fdmacc % 2
fdmacc = round(np.count_nonzero(K) / Mg) - 1
coeffs = coefficients(deriv=2, acc=fdmacc)["center"]["coefficients"]

ufem_0 = np.zeros(Mg)
ufem_1 = np.zeros(Mg)
ufem_2 = np.zeros(Mg)

ufdm_0 = np.zeros(Mg)
ufdm_1 = np.zeros(Mg)
ufdm_2 = np.zeros(Mg)

fig, ax = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
line0, = ax[0].plot(xfdmm, ufdm_0)
line1, = ax[1].plot(xfem, ufem_0)
ymm = 1e-12
ax[0].set_ylim(-ymm, ymm)
ax[0].set_xlabel(None)
ax[0].grid()
ax[1].set_ylim(-ymm, ymm)
ax[0].set_title(f"FDM acc = ${fdmacc}$")
ax[1].set_title(rf"FEM $p = {polyOrd}$")
ax[1].grid()

def update(mt: int):
    global ufem_0, ufem_1, ufem_2, ufdm_0, ufdm_1
    ufdm_1, ufdm_2 = ufdm_0, ufdm_1
    # lap[1:-1] = ufdm_1[:-2] - 2 * ufdm_1[1:-1] + ufdm_1[2:]
    lap = np.convolve(ufdm_1, coeffs, mode="same")
    ufdm_0 = 2 * ufdm_1 - ufdm_2 + c2fdmm * lap
    ufdm_0[Mg // 3] += dt**2 * s[mt] / dx

    ufem_1, ufem_2 = ufem_0, ufem_1
    ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * c2fem * Minv.T @ (f * s[mt] / (dxs[0] * c2fem) - K.T @ ufem_1)
    # ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * Minv.T @ (f * s[mt] / dxs[0] - c2fem * K.T @ ufem_1)
    
    if SAVE_MOV and not mt % 100:
        print(f"{100*(mt+1)/Mt:.1f}%")
    if SAVE_MOV or not mt % 10:
        fig.suptitle(f"{100*(mt+1)/Mt:.1f}%")
        line0.set_ydata(ufdm_0)
        line1.set_ydata(ufem_0)
        if not SAVE_MOV:
            plt.pause(0.00001)
    return line0, line1

# def update(nt):
#     lines[0].set_ydata(g_R(x, t[nt]))
#     lines[1].set_ydata(g_L(x, t[nt]))
#     title.set_text(f"t = {t[nt]:.1f} s")
#     return *lines, title

if SAVE_MOV:
    fps = 200
    ani = animation.FuncAnimation(fig, update, frames=Mt, blit=True)
    ani.save(f"fd_acc_{fdmacc}_fem_p_{polyOrd}.mp4", fps=fps)
else:
    for mt in tqdm(range(Mt)):
        update(mt)

# plt.show(block=True)