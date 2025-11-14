import numpy as np
import matplotlib.pyplot as plt

def initial_condition(x):
    return x * (3 * x**4 - 5 * x**3 + 2)

def implicit_step_neumann(u_old, dx, dt, Re, u_left, u_right):
    n = len(u_old) - 1 
    N = n - 1          

    A = np.zeros(N + 1)
    B = np.zeros(N + 1)
    C = np.zeros(N + 1)
    D = np.zeros(N + 1)

    for i in range(1, n):
        ui = u_old[i]

        A[i] = -1 / (2 * dx) - 1 / (Re * dx**2)
        B[i] = 1 / dt + 2 / (Re * dx**2)
        C[i] = 1 / (2 * dx) - 1 / (Re * dx**2)
        D[i] = u_old[i] / dt

    D[1] -= A[1] * u_left
    A[1] = 0.0  

    D[N] -= C[N] * u_right
    C[N] = 0.0  

    alpha = np.zeros(N + 1)
    beta = np.zeros(N + 1)

    alpha[1] = -C[1] / B[1]
    beta[1] = D[1] / B[1]

    for i in range(2, N + 1):
        denom = B[i] + A[i] * alpha[i - 1]
        alpha[i] = -C[i] / denom
        beta[i] = (D[i] - A[i] * beta[i - 1]) / denom

    u_new = np.zeros_like(u_old)
    u_new[0] = u_left
    u_new[n] = u_right

    u_new[N] = beta[N]
    for i in range(N - 1, 0, -1):
        u_new[i] = alpha[i] * u_new[i + 1] + beta[i]

    return u_new

def solve_pde_neumann(Re, L=1.0, n=100, dt=1e-3, t_end=0.1):
    dx = L / n
    x = np.linspace(0.0, L, n + 1)

    u = initial_condition(x)
    u_left = 0.0  
    u_right = 0.0  

    save_times = [0.0, 0.02, 0.05, t_end]
    save_times = sorted(list(set(t for t in save_times if t <= t_end + 1e-12)))

    profiles = {0.0: u.copy()}
    t = 0.0
    step = 0

    while t < t_end - 1e-12:
        u = implicit_step_neumann(u, dx, dt, Re, u_left, u_right)
        t += dt
        step += 1

        for ts in save_times:
            if abs(t - ts) < 0.5 * dt and ts not in profiles:
                profiles[ts] = u.copy()

    if t_end not in profiles:
        profiles[t_end] = u.copy()

    return x, profiles

Re_list = [1,50, 1000]  
colors = ['k', 'b', 'r', 'g', 'm']  

for Re in Re_list:
    x, profiles = solve_pde_neumann(Re, n=100, dt=1e-3, t_end=0.1)

    plt.figure(figsize=(8, 5))
    for j, (t, u) in enumerate(sorted(profiles.items())):
        plt.plot(x, u, label=f"t = {t:.3f}")
    plt.title(f"Neumann BC Solution, Re = {Re}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

plt.show()