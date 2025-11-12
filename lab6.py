import numpy as np
import matplotlib.pyplot as plt
from math import pi, ceil

# ---- basic setup ----
L, a = 2.0, 4.0                  # domain length, diffusivity
Nx   = 160                        # space steps (Nx+1 points)
T    = 0.10                       # final time
r0   = 0.45                       # target r<=0.5 for SIM
K    = 301                        # Fourier terms
dx   = L / Nx
x    = np.linspace(0.0, L, Nx+1)

# time step so we end exactly at T
dt = r0 * dx*dx / a
Nt = int(ceil(T/dt))
dt = T / Nt
r  = a * dt / (dx*dx)

show_t = [0.0, 0.01, 0.04, T]     # times to plot

# ---- initial condition ----
def f(x):
    return np.where(x <= 1.0, x, 2.0 - x)
u0 = f(x)

# ---- Fourier (analytic) ----
def b(n):                         # closed form coeffs
    if n % 2 == 0: return 0.0
    return 8.0/(n*n*pi*pi) * ((-1)**((n-1)//2))

def u_fourier(x, t, K=K):
    n = np.arange(1, K+1).reshape(-1,1)
    B = np.vectorize(b)(n).reshape(-1,1)
    S = np.sin(n * pi * x.reshape(1,-1) / 2.0)
    D = np.exp(-(n*n)*(pi*pi)*t)
    return (B*D*S).sum(axis=0)

# ---- Explicit (SIM) ----
def run_sim(u0):
    u = u0.copy(); u[0]=0.0; u[-1]=0.0
    snaps = {0.0: u.copy()}
    for n in range(1, Nt+1):
        v = u.copy()
        v[1:Nx] = u[1:Nx] + r*(u[2:Nx+1]-2*u[1:Nx]+u[0:Nx-1])
        v[0]=0.0; v[-1]=0.0
        u = v
        t = n*dt
        if any(abs(t-s)<1e-12 for s in show_t): snaps[round(t,8)] = u.copy()
    return u, snaps

# ---- Thomas tri-diagonal solver ----
def thomas(aL, aD, aU, d):
    n=len(d); c=np.zeros(n); e=np.zeros(n); x=np.zeros(n)
    c[0]=aU[0]/aD[0]; e[0]=d[0]/aD[0]
    for i in range(1,n):
        den = aD[i]-aL[i]*c[i-1]
        c[i] = (aU[i]/den) if i<n-1 else 0.0
        e[i] = (d[i]-aL[i]*e[i-1])/den
    x[-1]=e[-1]
    for i in range(n-2,-1,-1): x[i]=e[i]-c[i]*x[i+1]
    return x

# ---- Implicit (TA = Backward Euler + Thomas) ----
def run_ta(u0):
    u = u0.copy(); u[0]=0.0; u[-1]=0.0
    snaps = {0.0: u.copy()}
    nint = Nx-1
    aL = np.full(nint, -r); aD = np.full(nint, 1+2*r); aU = np.full(nint, -r)
    aL[0]=0.0; aU[-1]=0.0
    for n in range(1, Nt+1):
        d = u[1:Nx].copy()
        u[1:Nx] = thomas(aL, aD, aU, d)
        u[0]=0.0; u[-1]=0.0
        t = n*dt
        if any(abs(t-s)<1e-12 for s in show_t): snaps[round(t,8)] = u.copy()
    return u, snaps

# ---- run all ----
u_simT, sim = run_sim(u0)
u_taT,  ta  = run_ta(u0)
u_refT = u_fourier(x, T, K)

# helper: nearest stored snapshot to a target time
def snap(sdict, t):
    ks = np.array(list(sdict.keys()), float)
    k  = ks[np.argmin(abs(ks-t))]
    return sdict[k], float(k)

# errors at final time
e_sim = np.max(np.abs(snap(sim, T)[0] - u_refT))
e_ta  = np.max(np.abs(snap(ta,  T)[0] - u_refT))

print(f"Nx={Nx}, Nt={Nt}, dx={dx:.4f}, dt={dt:.6f}, r={r:.6f}")
print(f"max|SIM - Fourier| at T = {e_sim:.3e}")
print(f"max|TA  - Fourier| at T = {e_ta:.3e}")

# ---- plots ----
def fig1():
    plt.figure()
    for t in show_t:
        uf = u_fourier(x, t, K); us,_=snap(sim,t)
        plt.plot(x, uf, label=f"Fourier, t={t:.3f}")
        plt.plot(x, us, '--', label=f"SIM, t≈{snap(sim,t)[1]:.3f}")
    plt.title("Fourier vs SIM"); plt.xlabel("x"); plt.ylabel("U")
    plt.legend(); plt.grid(True, which='both', linewidth=0.5); plt.show()

def fig2():
    plt.figure()
    for t in show_t:
        uf = u_fourier(x, t, K); ui,_=snap(ta,t)
        plt.plot(x, uf, label=f"Fourier, t={t:.3f}")
        plt.plot(x, ui, '--', label=f"TA, t≈{snap(ta,t)[1]:.3f}")
    plt.title("Fourier vs TA"); plt.xlabel("x"); plt.ylabel("U")
    plt.legend(); plt.grid(True, which='both', linewidth=0.5); plt.show()

def fig3():
    plt.figure()
    us,_=snap(sim,T); ui,_=snap(ta,T)
    plt.plot(x, u_refT, label="Fourier (T)")
    plt.plot(x, us, '--', label="SIM (T)")
    plt.plot(x, ui, ':',  label="TA (T)")
    plt.title(f"Overlay at T={T}"); plt.xlabel("x"); plt.ylabel("U")
    plt.legend(); plt.grid(True, which='both', linewidth=0.5); plt.show()

fig1(); fig2(); fig3()
