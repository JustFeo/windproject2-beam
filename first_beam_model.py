import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

L = 100.0
EI = 1.0e10
mu = 5000.0
N_modes = 5

def char_eq(bL):
    return np.cosh(bL) * np.cos(bL) + 1.0

beta_L = []
beta_L.append(brentq(char_eq, 1.0, 2.5))
for n in range(2, N_modes + 1):
    guess = (2 * n - 1) * np.pi / 2
    beta_L.append(brentq(char_eq, guess - 0.5, guess + 0.5))

beta_L = np.array(beta_L)
beta = beta_L / L
omega = beta**2 * np.sqrt(EI / mu)

print("=" * 60)
print("PART 1 — Eigenvalues and natural frequencies")
print("=" * 60)
for n in range(N_modes):
    print(f"  Mode {n+1}:  betaL = {beta_L[n]:8.4f}   "
          f"omega = {omega[n]:10.4f} rad/s   "
          f"f = {omega[n] / (2 * np.pi):10.4f} Hz")


def mode_shape(x, bn, sigma_n):
    return (np.cosh(bn * x) - np.cos(bn * x)
            - sigma_n * (np.sinh(bn * x) - np.sin(bn * x)))

x = np.linspace(0, L, 500)
phi = np.zeros((N_modes, len(x)))

for n in range(N_modes):
    bn = beta[n]
    bL = beta_L[n]
    sigma = (np.cosh(bL) + np.cos(bL)) / (np.sinh(bL) + np.sin(bL))
    raw = mode_shape(x, bn, sigma)
    mass_int = np.trapezoid(raw**2, x) * mu
    phi[n] = raw / np.sqrt(mass_int)

fig, ax = plt.subplots(figsize=(9, 5))
for n in range(N_modes):
    ax.plot(x, phi[n], label=f"Mode {n+1}  (omega={omega[n]:.2f} rad/s)")
ax.set_xlabel("x  [m]")
ax.set_ylabel("phi_n(x)")
ax.set_title("Cantilever Eigenmodes — Fixed-Free Beam")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("eigenmodes.png")
print("\n→ Eigenmode plot saved to eigenmodes.png")

F0 = 1.0e5
Omega = omega[0] * 0.9
zeta = 0.02

print("\n" + "=" * 60)
print("PART 2 — Forced modal equations")
print("=" * 60)
print(f"  Driving: F0 = {F0:.0e} N  at tip,  Omega = {Omega:.4f} rad/s")
print(f"  Damping ratio zeta = {zeta}")

Qn0 = F0 * phi[:, -1]
print("\n  Modal forces Q_tilde_n_0 = F0 phi_n(L):")
for n in range(N_modes):
    print(f"    Mode {n+1}:  phi_n(L) = {phi[n, -1]:+.6f}   "
          f"Q_tilde_n_0 = {Qn0[n]:+.4e}")

H_n = np.zeros(N_modes)
theta = np.zeros(N_modes)
print("\n  Steady-state amplitudes:")
for n in range(N_modes):
    denom = np.sqrt((omega[n]**2 - Omega**2)**2 + (2 * zeta * omega[n] * Omega)**2)
    H_n[n] = Qn0[n] / denom
    theta[n] = np.arctan2(2 * zeta * omega[n] * Omega, omega[n]**2 - Omega**2)
    print(f"    Mode {n+1}:  H_n = {H_n[n]:+.6e}   theta_n = {np.degrees(theta[n]):+.2f} deg")


def modal_odes(t, y):
    eta = y[:N_modes]
    vel = y[N_modes:]
    deta = vel
    dvel = (-2 * zeta * omega * vel
            - omega**2 * eta
            + Qn0 * np.cos(Omega * t))
    return np.concatenate([deta, dvel])

T_period = 2 * np.pi / Omega
t_span = (0, 40 * T_period)
t_eval = np.linspace(*t_span, 4000)
y0 = np.zeros(2 * N_modes)

sol = solve_ivp(modal_odes, t_span, y0, t_eval=t_eval,
                method='RK45', rtol=1e-10, atol=1e-12)

w_tip = phi[:, -1] @ sol.y[:N_modes]
w_tip_ss = np.sum(
    H_n * phi[:, -1] * np.cos(Omega * sol.t[:, None] - theta),
    axis=1
)

fig2, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1 = axes[0]
ax1.plot(sol.t, w_tip, 'b-', lw=0.7, label='ODE integration')
ax1.plot(sol.t, w_tip_ss, 'r--', lw=0.7, alpha=0.7, label='Steady-state formula')
ax1.set_ylabel("w(L, t)  [m]")
ax1.set_title(f"Tip displacement — Omega/omega1 = {Omega/omega[0]:.2f},  zeta = {zeta}")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2 = axes[1]
for n in range(min(3, N_modes)):
    ax2.plot(sol.t, sol.y[n], lw=0.6, label=f"eta_{n+1}(t)")
ax2.set_xlabel("t  [s]")
ax2.set_ylabel("eta_n(t)")
ax2.set_title("Modal amplitudes (first 3 modes)")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("forced_response.png")
print("\n→ Forced-response plot saved to forced_response.png")

Omega_range = np.linspace(0.01 * omega[0], 3.0 * omega[0], 1000)
w_tip_amp = np.zeros_like(Omega_range)
for i, Om in enumerate(Omega_range):
    phasors = phi[:, -1] * Qn0 / ((omega**2 - Om**2) + 1j * 2 * zeta * omega * Om)
    w_tip_amp[i] = np.abs(np.sum(phasors))

fig3, ax3 = plt.subplots(figsize=(9, 5))
ax3.semilogy(Omega_range / omega[0], w_tip_amp, 'b-', lw=1.0)
for n in range(N_modes):
    if omega[n] <= Omega_range[-1]:
        ax3.axvline(omega[n] / omega[0], color='r', ls='--', lw=0.5, alpha=0.6)
        ax3.text(omega[n] / omega[0], ax3.get_ylim()[1] * 0.5,
                 f" omega_{n+1}", fontsize=8, color='r')
ax3.set_xlabel("Omega / omega1")
ax3.set_ylabel("|w(L)|  [m]")
ax3.set_title("Frequency Response — Tip Displacement Amplitude")
ax3.grid(True, alpha=0.3, which='both')
fig3.tight_layout()
fig3.savefig("frequency_response.png", dpi=150)
print("→ Frequency-response plot saved to frequency_response.png\n")
plt.show()
