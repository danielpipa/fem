import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import itertools
from tqdm import tqdm

c = [3000, 1500]
dt = 2e-9
dx = 20e-6
Lx = 10e-3
Lt = 5e-6
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

t0 = 3 * bw / f0
s = ss.gausspulse(t - t0, f0, bw)
def ricker(t, f0):
    sigma = .25 / f0
    tmp = (1 - (t/sigma)**2) * np.exp(-t**2 / (2 * sigma**2))
    return tmp / np.max(tmp)
s = ricker(t - t0, f0)
# plt.plot(t, s, t, s2)
# plt.show(block=True)

#%%

ufd_0 = np.zeros(Nx)
ufd_1 = np.zeros(Nx)
ufd_2 = np.zeros(Nx)

lap = np.zeros(Nx)

fig, ax = plt.subplots()
line, = ax.plot(x, ufd_0)
ymm = 1
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

def phi(x0, xi, i):
    N = len(xi)
    hb = xi[i] - xi[i-1] if i > 0 else -np.inf
    hf = xi[i+1] - xi[i] if i < N - 1 else -np.inf
    x = x0 - xi[i]
    conds = [(-hb <= x) & (x <= 0), (0 <= x) & (x <= hf)]
    funs = [lambda x: x/hb + 1, lambda x: 1 - x/hf, 0]
    return np.piecewise(x, conds, funs)

def dphi(x0, xi, i):
    N = len(xi)
    hb = xi[i] - xi[i-1] if i > 0 else -np.inf
    hf = xi[i+1] - xi[i] if i < N - 1 else -np.inf
    x = x0 - xi[i]
    conds = [(-hb <= x) & (x <= 0), (0 <= x) & (x <= hf)]
    funs = [lambda x: 1/hb, lambda x: -1/hf, 0]
    return np.piecewise(x, conds, funs)

dxfem = dx / 10
Nxfem = round(Lx / dxfem)
xfem = np.arange(Nxfem) * dxfem

# x = np.array([0,1,2,3,4,7,9,10])
# dxfem = .01
# xfem = np.arange(-3, 13, dxfem)
# for i in range(len(x)):
#     plt.plot(xfem, phi(xfem, x, i), color=f"C{i}")
#     plt.plot(xfem, dphi(xfem, x, i), color=f"C{i}", linestyle="dashed")
# plt.show(block=True)

def buildM(xi):
    N = len(xi)
    M = np.zeros((N, N))
    # for i in range(N):
    #     for j in range(N):
    for i, j in tqdm(itertools.product(range(N), range(N))):
        M[i, j] = np.sum(phi(xfem, xi, i) * phi(xfem, xi, j) * dxfem)
    return M

def buildK(xi):
    N = len(xi)
    K = np.zeros((N, N))
    # for i in range(N):
    #     for j in range(N):
    for i, j in tqdm(itertools.product(range(N), range(N))):
        K[i, j] = np.sum(dphi(xfem, xi, i) * dphi(xfem, xi, j) * dxfem)
    return K

def buildf(xi, k):
    N = len(xi)
    f = np.zeros(N)
    f[k] = np.sum(phi(xfem, xi, k) * dxfem)
    return f

def buildu(ufem):
    N = len(ufem)
    u = np.zeros(N)
    for n in range(N):
        u += ufem * phi(x, x, n)
    return u

M = buildM(x)
K = buildK(x)


Minv = np.linalg.inv(M)

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
f = buildf(x, 1)
ufem_0 = np.zeros(Nx)
ufem_1 = np.zeros(Nx)
ufem_2 = np.zeros(Nx)

fig, ax = plt.subplots()
line, = ax.plot(x, ufem_0)
ymm = 10
ax.set_ylim(-ymm, ymm)

for nt in range(Nt):
    ufem_1, ufem_2 = ufem_0, ufem_1
    # ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * Minv.T @ (f*s[nt] - c[0]**2 * K.T @ ufem_1)
    # ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * c[0] ** 2 * Minv.T @ (f * s[nt] - ufem_1)
    ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * ( - c[0] ** 2 * K.T @ ufem_1 / dx)
    ufem_0[Nx//2] += s[nt]
    line.set_ydata(buildu(ufem_0))
    # line.set_ydata(ufem_0)
    ax.set_title(f"{nt/Nt:.2f}")
    plt.pause(0.0001)