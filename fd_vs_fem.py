import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from tqdm import tqdm
import sympy as sym

cs = [6000, 1500]
dxs = [40e-6, 10e-6]
dt = .8e-9
Lx = 15e-3
Lt = 6e-6

xfem = np.block([np.arange(0, Lx/2, dxs[0]), np.arange(Lx/2, Lx + dxs[1], dxs[1])])
Nx = len(xfem)
dx = Lx / (Nx-1)
xfd = np.arange(Nx) * dx

Nt = round(Lt / dt)

C = max(cs) * dt / min(dxs)
print(f"Courant number: {C}")
if C > 1:
    raise ValueError(f"Courant number {C} > 1")

c2fd = np.zeros(Nx)
c2fd[xfd <= Lx/2] = cs[0]**2 * dt**2 / dx**2
c2fd[xfd > Lx/2] = cs[1]**2 * dt**2 / dx**2

c2fem = np.zeros(Nx)
c2fem[xfem <= Lx/2] = cs[0]**2
c2fem[xfem > Lx/2] = cs[1]**2

f0 = 5e6
# bw = .99
t = np.arange(Nt) * dt

t0 = 3 / f0
# s = ss.gausspulse(t - t0, f0, bw)
def ricker(t, f0):
    sigma = .25 / f0
    tmp = (1 - (t/sigma)**2) * np.exp(-t**2 / (2 * sigma**2))
    return tmp / np.max(tmp)
s = ricker(t - t0, f0)
# plt.plot(t, s)
# plt.show(block=True)

xsp = sym.Symbol("xsp")
xspim1, xspi, xspip1 = sym.symbols("xspim1, xspi, xspip1")
xspjm1, xspj, xspjp1 = sym.symbols("xspjm1, xspj, xspjp1")
phi_i = sym.Piecewise(((xsp - xspi) / (xspi - xspim1) + 1, sym.And(xspim1 <= xsp, xsp <= xspi)),
                      (1 - (xsp-xspi) / (xspip1-xspi), sym.And(xspi <= xsp, xsp <= xspip1)),
                      (0, True))

ksi = (xsp-xspi)/(xspip1-xspi)
N1 = sym.Matrix([sym.Piecewise((1 - ksi, sym.And(xspi <= xsp, xsp <= xspip1)), (0, True)),
               sym.Piecewise((ksi, sym.And(xspi <= xsp, xsp <= xspip1)), (0, True))])
dN1 = N1.diff(xsp)
N2 = sym.Matrix([sym.Piecewise((1 - 3*ksi + 2*ksi**2, sym.And(xspi <= xsp, xsp <= xspip1)), (0, True)),
               sym.Piecewise((4*ksi - 4*ksi**2, sym.And(xspi <= xsp, xsp <= xspip1)), (0, True)),
               sym.Piecewise((-ksi + 2*ksi**2, sym.And(xspi <= xsp, xsp <= xspip1)), (0, True))])
dN2 = N2.diff(xsp)

phi_j = phi_i.subs({xspim1: xspjm1, xspi: xspj, xspip1: xspjp1})
dphi_i = sym.diff(phi_i, xsp)
dphi_j = sym.diff(phi_j, xsp)
f_i = sym.Piecewise((1, sym.And(xspim1 <= xsp, xsp <= xspip1)), (0, True))

# Debug
DEBUG = True
# DEBUG = False

if DEBUG:
    # dxfem = dx / 1000
    # Nxfem = round(Lx / dxfem)
    # xfem = np.arange(Nxfem) * dxfem
    #
    xfem = np.array([0,1,2,3,4,7,9,10])
    # # x = np.arange(10)
    # N = len(xfem)
    # dxfem = .01
    # x = np.arange(-3, 13, dxfem)
    # for i in range(N):
    #     xim1 = xfem[i-1] if i > 0 else xfem[0]+1
    #     xip1 = xfem[i+1] if i < N - 1 else xfem[-1]-1
    #     xi = xfem[i]
    #     # plt.plot(xfem, sym.lambdify(xsp, phi_i.subs({xspim1: xim1, xspi: xi, xspip1: xip1}), "numpy")(xfem), color=f"C{i}")
    #     # plt.plot(xfem, sym.lambdify(xsp, dphi_i.subs({xspim1: xim1, xspi: xi, xspip1: xip1}), "numpy")(xfem), color=f"C{i}", linestyle="dashed")
    #     for n1 in N1:
    #         plt.plot(x, sym.lambdify(xsp, n1.subs({xspi: xi, xspip1: xip1}), "numpy")(x), color=f"C{i}")
    # plt.show(block=True)
    pass

def buildMat(x, phi_i, phi_j):
    N = len(x)
    M = np.zeros((N, N))
    Mij = sym.lambdify((xspim1, xspi, xspip1, xspjm1, xspj, xspjp1),
                       sym.integrate(phi_i * phi_j, (xsp, x[0], x[-1])), "numpy")
    for i in tqdm(range(N)):
        xim1 = x[i - 1] if i > 0 else x[0]+1
        xip1 = x[i + 1] if i < N - 1 else x[-1]-1
        xi = x[i]
        for j in range(max(0, i-1), min(N, i+2)):
            xjm1 = x[j - 1] if j > 0 else x[0]+1
            xjp1 = x[j + 1] if j < N - 1 else x[-1]-1
            xj = x[j]
            M[i, j] = Mij(xim1, xi, xip1, xjm1, xj, xjp1)
    return M

def buildf(x, xs):
    fi = sym.lambdify((xspim1, xspi, xspip1),
                      sym.integrate(f_i * phi_i, (xsp, x[0], x[-1])), "numpy")
    f = np.zeros(len(x))
    k = np.argmin(np.abs(xs - x))
    f[k] = fi(x[k-1], x[k], x[k+1])
    return f

# phii = sym.lambdify((xsp, xspim1, xspi, xspip1), phi_i, "numpy")
# def buildu(x, ufem):
#     N = len(ufem)
#     u = np.zeros(N)
#     for i in range(N):
#         xim1 = x[i - 1] if i > 0 else x[0]+1
#         xip1 = x[i + 1] if i < N - 1 else x[-1]-1
#         xi = x[i]
#         u += ufem[i] * phii(x, xim1, xi, xip1)
#     return u

def buildMatN(x, Ne):
    N = len(x)  # Number of nodes
    ndof = len(Ne)  # Number of degrees of freedom per element
    Ndof = len(x) * (ndof - 1)  # Number of degrees of freedom TOTAL
    M = np.zeros((Ndof, Ndof))
    Me_lmbd = sym.lambdify((xspi, xspip1), (Ne @ Ne.T).integrate((xsp, xspi, xspip1)), "numpy")
    for i in tqdm(range(N - 1)):
        xip1 = x[i + 1] if i < N - 1 else x[-1]-1
        xi = x[i]
        M[np.ix_(np.arange(i, i + ndof), np.arange(i, i + ndof))] += Me_lmbd(xi, xip1)
    return M

# def buildu(x, ui, Ne):
#     N = len(x)
#     u = np.zeros(N)
#     for ne in Ne:


print("Building matrices...")

M = buildMatN(xfem, N1)
K = buildMatN(xfem, dN1)

Minv = np.linalg.inv(M)

# M = sparray(M)
# K = sparray(K)
# Minv = sparray(Minv)

if DEBUG:
    M = buildMat(xfem, phi_i, phi_j)
    MN = buildMatN(xfem, N2)
    K = buildMat(xfem, dphi_i, dphi_j)
    KN = buildMatN(xfem, dN1)
    # print(f"K vs KN {np.sum(np.abs(K - KN))}")
    # print(f"M vs MN {np.sum(np.abs(M - MN))}")
    plt.figure()
    plt.imshow(K)
    plt.title("K")
    plt.figure()
    plt.imshow(KN)
    plt.title("KN")
    plt.figure()
    plt.imshow(M)
    plt.title("M")
    plt.figure()
    plt.imshow(MN)
    plt.title("MN")
    # plt.figure()
    # plt.imshow(Minv)
    # plt.title("Minv")
    plt.show(block=True)

f = buildf(xfem, Lx/3)

#%%
ufd_0 = np.zeros(Nx)
ufd_1 = np.zeros(Nx)
ufd_2 = np.zeros(Nx)

ufem_0 = np.zeros(Nx)
ufem_1 = np.zeros(Nx)
ufem_2 = np.zeros(Nx)

lap = np.zeros(Nx)
fig, ax = plt.subplots(2, 1)
line0, = ax[0].plot(xfd, ufd_0)
line1, = ax[1].plot(xfem, ufem_0)
ymm = 1e-12
ax[0].set_ylim(-ymm, ymm)
ax[0].set_xlabel(None)
ax[1].set_ylim(-ymm, ymm)

ax[0].set_title(f"FD")
ax[1].set_title(f"FEM")

plt.tight_layout()

for nt in tqdm(range(Nt)):
    ufd_1, ufd_2 = ufd_0, ufd_1
    lap[1:-1] = ufd_1[:-2] - 2 * ufd_1[1:-1] + ufd_1[2:]
    ufd_0 = 2 * ufd_1 - ufd_2 + c2fd * lap
    ufd_0[Nx//3] += dt**2 * s[nt] / dx

    ufem_1, ufem_2 = ufem_0, ufem_1
    ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * c2fem * Minv.T @ (f*s[nt]/(dxs[0]*c2fem) - K.T @ ufem_1)
    
    if not nt % 30:
        line0.set_ydata(ufd_0)
        # line1.set_ydata(buildu(xfem, ufem_0))
        line1.set_ydata(ufem_0)
    # ax[1].set_title(f"FEM {nt/Nt:.2f}")
    plt.pause(0.0001)
