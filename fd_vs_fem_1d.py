import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from tqdm import tqdm
import sympy as sym

# FEM polynomial order
polyOrd = 3

cs = [6000, 1500]  # Propagation speeds
dxs = [polyOrd * 80e-6, polyOrd * 20e-6]  # Spatial step
dt = .3e-9  # Time step
Lx = 15e-3  # Spatial length
Lt = 6e-6  # Temporal length
f0 = 5e6  # Central frenquency

# cs = [343.0, 343.0]  # Speed of sound (m/s)
# dxs = [.25, .25]
# Lx = 20.0  # Length of the domain (meters)
# Lt = 0.1  # Total simulation time (seconds)
# f0 = 200
#
# # Stability condition (CFL) for quadratic elements
# dt = 0.5 * (min(dxs) / max(cs)) / 2.0
# # dt = 10e-6

xfem = np.block([np.arange(0, Lx/2, dxs[0]), np.arange(Lx/2, Lx + dxs[1], dxs[1])])
xgrid = np.arange(0, Lx/2, dxs[0])
xgrid = np.append(xgrid, np.arange(xgrid[-1] + dxs[0], Lx + dxs[1], dxs[1]))

# xfem = np.arange(0, Lx/2, dxs[0] / polyOrd)
# xfem = np.append(xfem, np.arange(xfem[-1] + dxs[0] / polyOrd, Lx + dxs[1] / polyOrd, dxs[1] / polyOrd))

# xgrid = xfem[::polyOrd]
# xfem = xfem[:-1]


# xfem = xgrid[:-1]
# xfem = np.arange(0, Lx, dxs[0])
# Mx = len(xfem)  # Number of grid points
# Me = Mx - 1  # Number of elements
Me = len(xgrid) - 1
Mg = Me * polyOrd + 1  # Number of collocation points (global matrices)

def build_xfem():
    xfem = np.zeros(Mg)
    for i in range(Mg):
        if i % 2 == 0:
            xfem[i] = xgrid[i//polyOrd]
        else:
            xfem[i] = (xgrid[(i-1)//polyOrd] + xgrid[(i+1)//polyOrd]) / 2
    return xfem

xfem = build_xfem()

Mt = round(Lt / dt)  # Number of temporal points

C = max(cs) * dt / min(dxs)
print(f"Courant number: {C}")
if C > 1:
    raise ValueError(f"Courant number {C} > 1")

c2fem = np.zeros(Mg)
c2fem[xfem <= Lx/2] = cs[0]**2
c2fem[xfem > Lx/2] = cs[1]**2

# bw = .99
t = np.arange(Mt) * dt

t0 = 1.5 / f0
# s = ss.gausspulse(t - t0, f0, bw)
def ricker(t, f0):
    sigma = .25 / f0
    tmp = (1 - (t/sigma)**2) * np.exp(-t**2 / (2 * sigma**2))
    return tmp / np.max(tmp)
s = ricker(t - t0, f0)
# plt.plot(t, s)
# plt.show(block=True)

xsym = sym.Symbol("xsym")
xsymi, xsymip1, hsym = sym.symbols("xsymi, xsymip1, hsym")

xisym = (2 * xsym / hsym)

def buildShapeFunctions(p):
    """Lagrange polynomials"""
    Ne = sym.Matrix(p+1, 1, np.ones(p+1))
    xi = np.linspace(-1, 1, p+1)
    for i in range(p+1):
        for j in range(p+1):
            if i != j:
                Ne[i, 0] *= (xisym - xi[j]) / (xi[i] - xi[j])
    return Ne

Ne = buildShapeFunctions(polyOrd)
# if polyOrd == 1:
#     Ne = buildShapeFunctions(polyOrd)
# if polyOrd > 1:
#     # Ne = N2
#     c2fem = np.kron(c2fem, np.ones(polyOrd))#[:-1]

# h = 1
# x_ = np.arange(-h/2, h/2, .01)
# for ne in Ne:
#     plt.plot(x_, sym.lambdify((xsym, hsym), ne, "numpy")(x_, h))
# plt.show(block=True)

#%%

# f_i = sym.Piecewise((1, sym.And(xsymi <= xsym, xsym <= xsymip1)), (0, True))
f_i = sym.Piecewise((1, sym.And(-hsym <= xsym, xsym <= hsym)), (0, True))

def buildMat(Ne):
    M = np.zeros((Mg, Mg))
    Me_lmbd = sym.lambdify((hsym), (Ne @ Ne.T).integrate((xsym, -hsym/2, hsym/2)), "numpy")
    for i in tqdm(range(Me)):
        x_ip1 = xgrid[i + 1] if i < Me else xgrid[-1]-1
        x_i = xgrid[i]
        r = np.arange(polyOrd * i, polyOrd * i + polyOrd + 1)
        M[np.ix_(r, r)] += Me_lmbd(x_ip1 - x_i)
    return M

fe_lmbd = sym.lambdify((hsym), (Ne * f_i).integrate((xsym, -hsym/2, hsym/2)), "numpy")

def buildf(xs):
    f = np.zeros(Mg)
    k = np.argmin(np.abs(xs - xfem))
    f[k] = np.sum(fe_lmbd(xfem[k + polyOrd] - xfem[k]))
    return f

def build_u(ufem):
    u = np.zeros(Mg)

dNe = Ne.diff(xsym)

print("Building matrices...")
M = buildMat(Ne)
K = buildMat(dNe)
f = buildf(Lx/3)
Minv = np.linalg.inv(M)
print("Done building matrices.")

# M = sparray(M)
# K = sparray(K)
# Minv = sparray(Minv)

# Debug
# DEBUG_SHOW = True
DEBUG_SHOW = False

if DEBUG_SHOW:
    # M = buildMat(xfem, N2)
    # K = buildMat(xfem, dN2)
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

ufem_0 = np.zeros(Mg)
ufem_1 = np.zeros(Mg)
ufem_2 = np.zeros(Mg)

dx = Lx / (Mg - 1)
xfd = np.arange(Mg) * dx
c2fd = np.zeros(Mg)
c2fd[xfd <= Lx/2] = cs[0]**2 * dt**2 / dx**2
c2fd[xfd > Lx/2] = cs[1]**2 * dt**2 / dx**2

ufd_0 = np.zeros(Mg)
ufd_1 = np.zeros(Mg)
ufd_2 = np.zeros(Mg)

lap = np.zeros(Mg)
fig, ax = plt.subplots(2, 1)
line0, = ax[0].plot(xfd, ufd_0)
# line1, = ax[1].plot(xfem, ufem_0[::polyOrd])
line1, = ax[1].plot(xfem, ufem_0)
ymm = 1e-12
ax[0].set_ylim(-ymm, ymm)
ax[0].set_xlabel(None)
ax[1].set_ylim(-ymm, ymm)

ax[0].set_title(f"FD")
ax[1].set_title(rf"FEM $p = {polyOrd}$")

plt.tight_layout()

for nt in tqdm(range(Mt)):
    ufd_1, ufd_2 = ufd_0, ufd_1
    lap[1:-1] = ufd_1[:-2] - 2 * ufd_1[1:-1] + ufd_1[2:]
    ufd_0 = 2 * ufd_1 - ufd_2 + c2fd * lap
    ufd_0[Mg // 3] += dt**2 * s[nt] / dx

    ufem_1, ufem_2 = ufem_0, ufem_1
    ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * c2fem * Minv.T @ (f*s[nt]/(dxs[0]*c2fem) - K.T @ ufem_1)
    # ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * Minv.T @ (f * s[nt] / dxs[0] - c2fem * K.T @ ufem_1)
    
    if not nt % 100:
        line0.set_ydata(ufd_0)
        line1.set_ydata(ufem_0)
    plt.pause(0.0001)
