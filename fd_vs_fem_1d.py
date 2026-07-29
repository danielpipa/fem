import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from tqdm import tqdm
import sympy as sym
import chaospy
from numpy.polynomial import legendre

# FEM polynomial order of shape function
polyOrd = 5

cs = [6000, 1500]  # Propagation speeds
dxs = [polyOrd * 100e-6, polyOrd * 30e-6]  # Spatial step
dt = .3e-9  # Time step
Lx = 15e-3  # Spatial length
Lt = 6e-6  # Temporal length
f0 = 5e6  # Central frenquency

# Grid points
xgrid = np.arange(0, Lx/2, dxs[0])
xgrid = np.append(xgrid, np.arange(xgrid[-1] + dxs[0], Lx + dxs[1], dxs[1]))

Mx = len(xgrid)  # Number of grid points
Me = Mx - 1  # Number of elements
Mg = Me * polyOrd + 1  # Number of collocation points (global matrices)


def gll_nodes_and_weights(n):
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
def ricker(t, f0):
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

def buildShapeFunctions(p):
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

def buildMat(Ne):
    """Generic function to build the mass and the stiffness matrix"""
    M = np.zeros((Mg, Mg))
    Me_lmbd = sym.lambdify((hsym), (Ne @ Ne.T).integrate((xsym, -hsym/2, hsym/2)), "numpy")
    for i in tqdm(range(Me)):
        x_ip1 = xgrid[i + 1] if i < Me else xgrid[-1]-1
        x_i = xgrid[i]
        r = np.arange(polyOrd * i, polyOrd * i + polyOrd + 1)
        M[np.ix_(r, r)] += Me_lmbd(x_ip1 - x_i)
    return M


def buildf(xs):
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
# TODO: increase derivative accuracy of fdmM with polynomial order for fairer comparison

ufem_0 = np.zeros(Mg)
ufem_1 = np.zeros(Mg)
ufem_2 = np.zeros(Mg)

ufdmm_0 = np.zeros(Mg)
ufdmm_1 = np.zeros(Mg)
ufdmm_2 = np.zeros(Mg)

lap = np.zeros(Mg)
fig, ax = plt.subplots(2, 1)
line0, = ax[0].plot(xfdmm, ufdmm_0)
line1, = ax[1].plot(xfem, ufem_0)
ymm = 1e-12
ax[0].set_ylim(-ymm, ymm)
ax[0].set_xlabel(None)
ax[0].grid()
ax[1].set_ylim(-ymm, ymm)
ax[0].set_title(f"fdmM")
ax[1].set_title(rf"FEM $p = {polyOrd}$")
ax[1].grid()

plt.tight_layout()

for nt in tqdm(range(Mt)):
    ufdmm_1, ufdmm_2 = ufdmm_0, ufdmm_1
    lap[1:-1] = ufdmm_1[:-2] - 2 * ufdmm_1[1:-1] + ufdmm_1[2:]
    ufdmm_0 = 2 * ufdmm_1 - ufdmm_2 + c2fdmm * lap
    ufdmm_0[Mg // 3] += dt**2 * s[nt] / dx

    ufem_1, ufem_2 = ufem_0, ufem_1
    ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * c2fem * Minv.T @ (f*s[nt]/(dxs[0]*c2fem) - K.T @ ufem_1)
    # ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * Minv.T @ (f * s[nt] / dxs[0] - c2fem * K.T @ ufem_1)
    
    if not nt % 100:
        line0.set_ydata(ufdmm_0)
        line1.set_ydata(ufem_0)
    plt.pause(0.0001)

plt.show(block=True)