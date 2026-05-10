import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq

L = 30.0
EI = 1.0e9
mu = 500.0
F0 = 1.0e3
Omega = 0.5
zeta = 0.02
N_MODES = 6


def char_eq_lambda(lam):
    return np.cosh(lam) * np.cos(lam) + 1.0


def find_lambda_roots(n_modes):
    roots = []
    for n in range(1, n_modes + 1):
        a = (2 * n - 1) * np.pi / 2.0 + 1.0e-6
        b = (2 * n + 1) * np.pi / 2.0 - 1.0e-6
        roots.append(brentq(char_eq_lambda, a, b))
    return np.array(roots)


def mode_shape_raw(x, beta):
    lam = beta * L
    c = (np.sinh(lam) - np.sin(lam)) / (np.cosh(lam) + np.cos(lam))
    return np.cosh(beta * x) - np.cos(beta * x) - c * (np.sinh(beta * x) - np.sin(beta * x))


def normalization_factor(beta):
    integrand = lambda xx: mode_shape_raw(xx, beta) ** 2
    norm_sq, _ = quad(integrand, 0.0, L, limit=200)
    return np.sqrt(norm_sq)


def build_modes(n_modes):
    lam = find_lambda_roots(n_modes)
    beta = lam / L
    omega = beta**2 * np.sqrt(EI / mu)
    norms = np.array([normalization_factor(b) for b in beta])

    def Wn(n, x):
        return mode_shape_raw(x, beta[n]) / norms[n]

    return omega, Wn


def main():
    omega_n, Wn = build_modes(N_MODES)
    wnL = np.array([Wn(i, L) for i in range(N_MODES)])
    t = np.linspace(0.0, 80.0, 2000)

    drive = (F0 / mu) * wnL
    den = (omega_n**2 - Omega**2) ** 2 + (2.0 * zeta * omega_n * Omega) ** 2
    a_cos = drive * (omega_n**2 - Omega**2) / den
    a_sin = drive * (2.0 * zeta * omega_n * Omega) / den
    eta = a_cos[:, None] * np.cos(Omega * t) + a_sin[:, None] * np.sin(Omega * t)
    w_tip = np.sum(eta * wnL[:, None], axis=0)

    print("First 3 natural frequencies [rad/s]:", omega_n[:3])
    print("Tip amplitude [m]:", np.max(np.abs(w_tip)))
    plt.plot(t, w_tip)
    plt.xlabel("t [s]")
    plt.ylabel("w(L,t) [m]")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
