import numpy as np
from scipy.optimize import brentq
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
print("Eigenvalues and natural frequencies")
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
ax.set_title("Cantilever Eigenmodes: Fixed-Free Beam")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("eigenmodes.png")
print("\n-> Eigenmode plot saved to eigenmodes.png")

plt.show()
