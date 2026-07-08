import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.integrate import quad, simpson
import itertools
from tqdm import tqdm
import sympy as sp # import symbols, Piecewise, integrate, And

c = [3000, 1500]
dt = 2e-9
dx = 20e-6
Lx = 5e-3
Lt = 2e-6
Nx = round(Lx / dx)
Nt = round(Lt / dt)

C = max(c) * dt / dx
print(f"Courant number: {C}")
if C > 1:
    raise ValueError(f"Courant number {C} > 1")
cc = np.zeros(Nx) # * c[0]**2 * dt**2 / dx**2
cc[:Nx//2] = c[0]
cc[Nx//2:] = c[1]
cc[:] = c[0]
cc = cc ** 2 * dt**2 / dx**2

f0 = 5e6
bw = .99

t = np.arange(Nt) * dt
x = np.arange(Nx) * dx

N = len(x)

t0 = 3 * bw / f0
s = ss.gausspulse(t - t0, f0, bw)
def ricker(t, f0):
    sigma = .25 / f0
    tmp = (1 - (t/sigma)**2) * np.exp(-t**2 / (2 * sigma**2))
    return tmp / np.max(tmp)
s = ricker(t - t0, f0)
# plt.plot(t, s, t, s2)
# plt.show(block=True)

# def phi(x0, xi, i):
#     hb = xi[i] - xi[i-1] if i > 0 else -np.inf
#     hf = xi[i+1] - xi[i] if i < N - 1 else -np.inf
#     x = x0 - xi[i]
#     conds = [(-hb <= x) & (x <= 0), (0 <= x) & (x <= hf)]
#     funs = [lambda x: x/hb + 1, lambda x: 1 - x/hf, 0]
#     return np.piecewise(x, conds, funs)
#
# def dphi(x0, xi, i):
#     hb = xi[i] - xi[i-1] if i > 0 else -np.inf
#     hf = xi[i+1] - xi[i] if i < N - 1 else -np.inf
#     x = x0 - xi[i]
#     conds = [(-hb <= x) & (x <= 0), (0 <= x) & (x <= hf)]
#     funs = [lambda x: 1/hb, lambda x: -1/hf, 0]
#     return np.piecewise(x, conds, funs)

xsp = sp.Symbol("xsp")
xspim1, xspi, xspip1 = sp.symbols("xspim1, xspi, xspip1")
xspjm1, xspj, xspjp1 = sp.symbols("xspjm1, xspj, xspjp1")
# phi = sp.Piecewise((xsp/hbsp + 1, sp.And(-hbsp <= xsp, xsp <= 0)), (1 - xsp/hfsp, sp.And(0 <= xsp, xsp <= hfsp)), (0, True))
phi_i = sp.Piecewise(((xsp-xspi)/(xspi-xspim1) + 1, sp.And(xspim1<=xsp, xsp<=xspi)),
                   (1 - (xsp-xspi)/(xspip1-xspi), sp.And(xspi<=xsp, xsp<=xspip1)),
                   (0, True))
phi_j = sp.Piecewise(((xsp-xspj)/(xspj-xspjm1) + 1, sp.And(xspjm1<=xsp, xsp<=xspj)),
                   (1 - (xsp-xspj)/(xspjp1-xspj), sp.And(xspj<=xsp, xsp<=xspjp1)),
                   (0, True))
dphi_i = sp.diff(phi_i, xsp)
dphi_j = sp.diff(phi_j, xsp)
f_i = sp.Piecewise((1, sp.And(xspim1<=xsp, xsp<=xspip1)), (0, True))

# dxfem = dx / 1000
# Nxfem = round(Lx / dxfem)
# xfem = np.arange(Nxfem) * dxfem
#
# # x = np.array([0,1,2,3,4,7,9,10])
# x = np.arange(10)
# N = len(x)
# dxfem = .01
# xfem = np.arange(-3, 13, dxfem)
# for i in range(N):
#     xim1 = x[i-1] if i > 0 else x[0]+1
#     xip1 = x[i+1] if i < N - 1 else x[-1]-1
#     xi = x[i]
#     plt.plot(xfem, sp.lambdify(xsp, phi_i.subs({xspim1: xim1, xspi: xi, xspip1: xip1}), "numpy")(xfem), color=f"C{i}")
#     plt.plot(xfem, sp.lambdify(xsp, dphi_i.subs({xspim1: xim1, xspi: xi, xspip1: xip1}), "numpy")(xfem), color=f"C{i}", linestyle="dashed")
# plt.show(block=True)

#%%

def buildMat(x, phi_i, phi_j):
    N = len(x)
    M = np.zeros((N, N))
    Mij = sp.lambdify((xspim1, xspi, xspip1, xspjm1, xspj, xspjp1), sp.integrate(phi_i * phi_j, (xsp, x[0], x[-1])), "numpy")
    for i, j in tqdm(itertools.product(range(N), range(N))):
        xim1 = x[i - 1] if i > 0 else x[0]+1
        xip1 = x[i + 1] if i < N - 1 else x[-1]-1
        xi = x[i]
        xjm1 = x[j - 1] if j > 0 else x[0]+1
        xjp1 = x[j + 1] if j < N - 1 else x[-1]-1
        xj = x[j]
        M[i, j] = Mij(xim1, xi, xip1, xjm1, xj, xjp1)
    return M

# def buildK(x):
#     K = np.zeros((N, N))
#     for i, j in tqdm(itertools.product(range(N), range(N))):
#         xim1 = x[i - 1] if i > 0 else np.inf
#         xip1 = x[i + 1] if i < N - 1 else -np.inf
#         xi = x[i]
#         xjm1 = x[j - 1] if j > 0 else np.inf
#         xjp1 = x[j + 1] if j < N - 1 else -np.inf
#         xj = x[j]
#         dphi_i = dphi_i.subs({xspim1: xim1, xspi: xi, xspip1: xip1})
#         dphi_j = dphi_i.subs({xspim1: xjm1, xspi: xj, xspip1: xjp1})
#         K[i, j] = sp.integrate(dphi_i * dphi_j, (xsp, x[0], x[-1]))
#     return K

def buildf(x, k):
    fi = sp.lambdify((xspim1, xspi, xspip1), sp.integrate(f_i * phi_i, (xsp, x[0], x[-1])), "numpy")
    f = np.zeros(len(x))
    f[k] = fi(x[k-1], x[k], x[k+1])
    return f

def buildu(ufem):
    N = len(ufem)
    u = np.zeros(N)
    for n in range(N):
        u += ufem * phi(x, x, n)
    return u

print("Building matrices...")
M = buildMat(x, phi_i, phi_j)
K = buildMat(x, dphi_i, dphi_j)

Minv = np.linalg.pinv(M)

plt.figure()
plt.imshow(K)
plt.title("K")
plt.figure()
plt.imshow(M)
plt.title("M")
plt.figure()
plt.imshow(Minv)
plt.title("Minv")
plt.show(block=True)


#%%
ufd_0 = np.zeros(Nx)
ufd_1 = np.zeros(Nx)
ufd_2 = np.zeros(Nx)

lap = np.zeros(Nx)
fig, ax = plt.subplots()
line, = ax.plot(x, ufd_0)
ymm = 10
ax.set_ylim(-ymm, ymm)

for nt in range(Nt):
    ufd_1, ufd_2 = ufd_0, ufd_1
    lap[1:-1] = ufd_1[:-2] - 2 * ufd_1[1:-1] + ufd_1[2:]
    ufd_0 = 2 * ufd_1 - ufd_2 + cc * lap
    ufd_0[1] += s[nt]
    line.set_ydata(ufd_0)
    ax.set_title(f"{nt/Nt:.2f}")
    plt.pause(0.0001)

#%%
f = buildf(x, 1)
ufem_0 = np.zeros(Nx)
ufem_1 = np.zeros(Nx)
ufem_2 = np.zeros(Nx)

fig, ax = plt.subplots()
line, = ax.plot(x, ufem_0)
ymm = 1e-16
ax.set_ylim(-ymm, ymm)

for nt in range(Nt):
    ufem_1, ufem_2 = ufem_0, ufem_1
    ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * Minv.T @ (f*s[nt] - c[0]**2 * K.T @ ufem_1)
    # ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * c[0] ** 2 * Minv.T @ (f * s[nt] - ufem_1)
    # ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * ( - c[0] ** 2 * K.T @ ufem_1 / dx)
    # ufem_0[Nx//2] += s[nt]
    # line.set_ydata(buildu(ufem_0))
    line.set_ydata(ufem_0)
    ax.set_title(f"{nt/Nt:.2f}")
    plt.pause(0.0001)