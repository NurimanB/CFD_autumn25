import numpy as np
import matplotlib.pyplot as plt

# data on [0,2]
L = 2.0
def f(x): return 2.0*x**2*(3.0 - x)

# cosine-series coeffs for Neumann on [0,2]
def a(n):
    return 192.0 * (((-1)**n) - 1.0) / (np.pi**4 * n**4)

# partial sum for u(x,0)
def s0(x, m):
    u = np.full_like(x, 4.0, dtype=float)       # A0
    for n in range(1, m+1):
        u += a(n) * np.cos(n*np.pi*x/L)
    return u

# solution u(x,t)
def u(x, t, m, k=1.0):
    u = np.full_like(x, 4.0, dtype=float)
    for n in range(1, m+1):
        lam2 = (n*np.pi/L)**2
        u += a(n) * np.cos(n*np.pi*x/L) * np.exp(-k*lam2*t)
    return u

# grid
x = np.linspace(0.0, L, 1000)

# Plot 1: f vs partial sums (M=5,10,20)
plt.figure(figsize=(10,6))
plt.plot(x, f(x), label='Initial f(x)=2x²(3−x)', linewidth=2)
for M, ls in [(5,'--'), (10,'-.'), (20,':')]:
    plt.plot(x, s0(x, M), ls, label=f'Fourier Approx. ({M} terms)')
plt.title('Fourier Series Approximation of f(x) on [0,2]')
plt.xlabel('x'); plt.ylabel('f and partial sums')
plt.grid(True); plt.legend(); plt.show()

# Plot 2: u(x,t) snapshots
plt.figure(figsize=(10,6))
for t in [0.0, 0.01, 0.05, 0.1]:
    plt.plot(x, u(x, t, m=80), label=f't={t}')  # more terms for smooth curves
plt.title('Solution to the Heat Equation u(x,t) with Fourier Series Approximation')
plt.xlabel('x'); plt.ylabel('u(x,t)')
plt.grid(True); plt.legend(); plt.show()
