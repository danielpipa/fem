import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from tqdm import tqdm
import sympy as sym

# FEM polynomial order
p = 1
# p = 2

cs = [6000, 1500]
dxs = [40e-6, 10e-6]
# dxs = [80e-6, 20e-6]
dt = .3e-9
Lx = 15e-3
Lt = 6e-6
f0 = 5e6

# cs = [343.0, 343.0]  # Speed of sound (m/s)
# dxs = [.25, .25]
# Lx = 20.0  # Length of the domain (meters)
# Lt = 0.1  # Total simulation time (seconds)
# f0 = 200
#
# # Stability condition (CFL) for quadratic elements
# dt = 0.5 * (min(dxs) / max(cs)) / 2.0
# # dt = 10e-6

# xfem = np.block([np.arange(0, Lx/2, dxs[0]), np.arange(Lx/2, Lx + dxs[1], dxs[1])])
xfem = np.arange(0, Lx/2, dxs[0])
xfem = np.append(xfem, np.arange(xfem[-1] + dxs[0], Lx + dxs[1], dxs[1]))
# xfem = np.arange(0, Lx, dxs[0])
Nx = len(xfem)

Nt = round(Lt / dt)

C = max(cs) * dt / min(dxs)
print(f"Courant number: {C}")
if C > 1:
    raise ValueError(f"Courant number {C} > 1")

c2fem = np.zeros(Nx)
c2fem[xfem <= Lx/2] = cs[0]**2
c2fem[xfem > Lx/2] = cs[1]**2

# bw = .99
t = np.arange(Nt) * dt

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

# ksi = (xsym-xsymi)/(xsymip1-xsymi)
ksi = (2*xsym/hsym)
# N1 = sym.Matrix([sym.Piecewise((1 - ksi, sym.And(xsymi <= xsym, xsym <= xsymip1)), (0, True)),
#                sym.Piecewise((ksi, sym.And(xsymi <= xsym, xsym <= xsymip1)), (0, True))])

N1 = sym.Matrix([.5*(1-ksi), .5*(1+ksi)])

# N2 = sym.Matrix([sym.Piecewise((1 - 3*ksi + 2*ksi**2, sym.And(xsymi < xsym, xsym < xsymip1)), (0, True)),
#                sym.Piecewise((4*ksi - 4*ksi**2, sym.And(xsymi < xsym, xsym < xsymip1)), (0, True)),
#                sym.Piecewise((-ksi + 2*ksi**2, sym.And(xsymi < xsym, xsym < xsymip1)), (0, True))])
#
# N2 = sym.Matrix([1 - 3*ksi + 2*ksi**2, 4*ksi - 4*ksi**2, -ksi + 2*ksi**2])

# N2 = sym.Matrix([sym.Piecewise((.5*ksi*(ksi-1), sym.And(xsymi <= xsym, xsym <= xsymip1)), (0, True)),
#                sym.Piecewise((1-ksi**2, sym.And(xsymi <= xsym, xsym <= xsymip1)), (0, True)),
#                sym.Piecewise((.5*ksi*(ksi+1), sym.And(xsymi <= xsym, xsym <= xsymip1)), (0, True))])

N2 = sym.Matrix([.5*ksi*(ksi-1), 1-ksi**2, .5*ksi*(ksi+1)])
# N2 = sym.Matrix([.5*(2*xsym/hsym)*(2*xsym/hsym-1), 1-(2*xsym/hsym)**2, .5*(2*xsym/hsym)*(2*xsym/hsym+1)])
# N2 = sym.Matrix([(xsym/hsym)*(2*xsym/hsym-1), 1-(2*xsym/hsym)**2, (xsym/hsym)*(2*xsym/hsym+1)])
# N2 = sym.Matrix([ksi*(ksi-1), 2*(1-ksi**2), ksi*(ksi+1)])

if p == 1:
    Ne = N1
elif p == 2:
    Ne = N2
    c2fem = np.kron(c2fem, np.ones(p))[:-1]

h = 2
x_ = np.arange(-h/2, h/2, .01)
for ne in Ne:
    plt.plot(x_, sym.lambdify((xsym, hsym), ne, "numpy")(x_, h))
plt.show(block=True)

#%%

# f_i = sym.Piecewise((1, sym.And(xsymi <= xsym, xsym <= xsymip1)), (0, True))
f_i = sym.Piecewise((1, sym.And(-hsym <= xsym, xsym <= hsym)), (0, True))

# Debug
# DEBUG1 = True
DEBUG1 = False

DEBUG_SHOW = True
# DEBUG_SHOW = False

if DEBUG1:
    dxfem = dx / 1000
    Nxfem = round(Lx / dxfem)
    xfem = np.arange(Nxfem) * dxfem

    xfem = np.array([0,1,2,3,4,7,9,10])
    xfem = np.array([-2, -1, 1, 2, 3])
    # x = np.arange(10)
    N = len(xfem)
    dxfem = .001
    x = np.arange(-3, 13, dxfem)
    for i in range(N):
        xim1 = xfem[i-1] if i > 0 else xfem[0]+1
        xip1 = xfem[i+1] if i < N - 1 else xfem[-1]-1
        xi = xfem[i]
        for n1 in N2:
            plt.plot(x, sym.lambdify(xsym, n1.subs({xsymi: xi, xsymip1: xip1}), "numpy")(x), color=f"C{i}")
    plt.show(block=True)
    pass

def buildMat(x, Ne):
    N = len(x)  # Number of nodes
    ndof = len(Ne)  # Number of degrees of freedom per element
    Ndof = (N - 1) * (ndof - 1) + 1  # Number of degrees of freedom TOTAL
    M = np.zeros((Ndof, Ndof))
    # Me_lmbd = sym.lambdify((xsymi, xsymip1), (Ne @ Ne.T).integrate((xsym, xsymi, xsymip1)), "numpy")
    Me_lmbd = sym.lambdify((hsym), (Ne @ Ne.T).integrate((xsym, -hsym, hsym)), "numpy")
    for i in tqdm(range(N - 1)):
        xip1 = x[i + 1] if i < N - 1 else x[-1]-1
        xi = x[i]
        r = np.arange((ndof - 1) * i, (ndof - 1) * i + ndof)
        # M[np.ix_(r, r)] += Me_lmbd(xi, xip1)
        # M[np.ix_(r, r)] += Me_lmbd(0, xip1 - xi)
        M[np.ix_(r, r)] += Me_lmbd(xip1 - xi)
    return M

def buildf(x, Ne, xs):
    N = len(x)  # Number of nodes
    ndof = len(Ne)  # Number of degrees of freedom per element
    Ndof = (N - 1) * (ndof - 1) + 1  # Number of degrees of freedom TOTAL
    # fe_lmbd = sym.lambdify((xsymi, xsymip1),
    #                   (Ne * f_i).integrate((xsym, xsymi, xsymip1)), "numpy")
    fe_lmbd = sym.lambdify((hsym),
                      (Ne * f_i).integrate((xsym, -hsym, hsym)), "numpy")
    f = np.zeros(Ndof)
    k = p * np.argmin(np.abs(xs - x))
    f[k] = np.sum(fe_lmbd(x[k+1]-x[k]))
    return f

# def buildu(x, ui, Ne):
#     N = len(x)
#     u = np.zeros(N)
#     for ne in Ne:

dNe = Ne.diff(xsym)

print("Building matrices...")
M = buildMat(xfem, Ne)
K = buildMat(xfem, dNe)
f = buildf(xfem, Ne, Lx/3)
Minv = np.linalg.inv(M)
print("Done building matrices.")

# M = sparray(M)
# K = sparray(K)
# Minv = sparray(Minv)

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

Ndof = K.shape[0]
ufem_0 = np.zeros(Ndof)
ufem_1 = np.zeros(Ndof)
ufem_2 = np.zeros(Ndof)

dx = Lx / (Ndof-1)
xfd = np.arange(Ndof) * dx
c2fd = np.zeros(Ndof)
c2fd[xfd <= Lx/2] = cs[0]**2 * dt**2 / dx**2
c2fd[xfd > Lx/2] = cs[1]**2 * dt**2 / dx**2

ufd_0 = np.zeros(Ndof)
ufd_1 = np.zeros(Ndof)
ufd_2 = np.zeros(Ndof)

lap = np.zeros(Ndof)
fig, ax = plt.subplots(2, 1)
line0, = ax[0].plot(xfd, ufd_0)
line1, = ax[1].plot(xfem, ufem_0[::p])
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
    ufd_0[Ndof//3] += dt**2 * s[nt] / dx

    ufem_1, ufem_2 = ufem_0, ufem_1
    ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * c2fem * Minv.T @ (f*s[nt]/(dxs[0]*c2fem) - K.T @ ufem_1)
    # ufem_0 = 2 * ufem_1 - ufem_2 + dt**2 * Minv.T @ (f * s[nt] / dxs[0] - c2fem * K.T @ ufem_1)
    
    if not nt % 100:
        line0.set_ydata(ufd_0)
        # line1.set_ydata(buildu(xfem, ufem_0))
        line1.set_ydata(ufem_0[::p])
    # ax[1].set_title(f"FEM {nt/Nt:.2f}")
    plt.pause(0.0001)
