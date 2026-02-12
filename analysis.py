# %%

import numpy as np
import matplotlib.pyplot as plt

def simulate_damped_driven_noisy_oscillator(
    m: float = 1.0,
    omega0: float = 2.0 * np.pi * 5.0,   # rad/s
    gamma: float = 2.0,                  # 1/s (damping rate)
    F0: float = 1.0,                     # driving force amplitude
    omega_d: float = 2.0 * np.pi * 5.0,   # driving frequency (rad/s)
    phi: float = 0.0,                    # drive phase
    noise_sigma: float = 0.5,            # noise strength (force std in N*sqrt(s))
    x0: float = 0.0,
    v0: float = 0.0,
    t_max: float = 2.0,
    dt: float = 1e-4,
    seed: int = 1,
):
    """
    Langevin model:
        dx = v dt
        dv = [-(2*gamma) v - omega0^2 x + (F0/m) cos(omega_d t + phi)] dt + (noise_sigma/m) dW

    Here dW ~ N(0, dt). So the stochastic term is (noise_sigma/m) * sqrt(dt) * N(0,1).
    noise_sigma has units of force * sqrt(time) so that noise term is a force.
    """
    rng = np.random.default_rng(seed)

    n = int(np.floor(t_max / dt)) + 1
    t = np.linspace(0.0, t_max, n)

    x = np.empty(n, dtype=float)
    v = np.empty(n, dtype=float)
    x[0], v[0] = x0, v0

    # Pre-generate Wiener increments
    dW = np.sqrt(dt) * rng.standard_normal(n - 1)

    for i in range(n - 1):
        ti = t[i]
        drive = (F0 / m) * np.cos(omega_d * ti + phi)

        # Drift
        a = - (2.0 * gamma) * v[i] - (omega0 ** 2) * x[i] + drive

        # Euler–Maruyama update
        x[i + 1] = x[i] + v[i] * dt
        v[i + 1] = v[i] + a * dt + (noise_sigma / m) * dW[i]

    return t, x, v

# %%

t, x, v = simulate_damped_driven_noisy_oscillator(
    omega0=2*np.pi*10,   # 10 Hz
    omega_d=2*np.pi*9.5, # near-resonant drive
    gamma=3.0,
    F0=50.0,
    noise_sigma=2.0,
    t_max=2.0,
    dt=1e-4,
    x0=0.0,
    v0=0.0,
    seed=3,
)

# Quick diagnostics: estimate steady-state amplitude via RMS over last 20%
tail = slice(int(0.8 * len(t)), None)
x_rms = np.sqrt(np.mean(x[tail] ** 2))

print(f"RMS displacement over last 20%: {x_rms:.4g}")

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
ax[0].plot(t, x)
ax[0].set_ylabel("x(t)")
ax[0].set_title("Damped driven oscillator with Langevin noise")

ax[1].plot(t, v)
ax[1].set_ylabel("v(t)")
ax[1].set_xlabel("time (s)")

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def gaussian_pulse(t, t0, sigma, area=1.0):
    """
    Gaussian pulse with specified integral (area) over time:
        F(t) = area * (1/(sqrt(2pi)*sigma)) * exp(-(t-t0)^2/(2 sigma^2))
    """
    return area * (1.0 / (np.sqrt(2.0*np.pi) * sigma)) * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

def simulate_force_to_displacement(
    t, F,
    m=1.0,
    omega0=2*np.pi*10,
    gamma=3.0,
    x0=0.0, v0=0.0,
    noise_sigma=0.0,  # set >0 for Langevin force noise (optional)
    seed=0
):
    """
    Simulate: x'' + 2 gamma x' + omega0^2 x = (1/m) F(t) + (noise_sigma/m) * xi(t)
    Using Euler–Maruyama for optional noise.
    """
    dt = t[1] - t[0]
    n = len(t)
    rng = np.random.default_rng(seed)

    x = np.empty(n)
    v = np.empty(n)
    x[0], v[0] = x0, v0

    # Wiener increments for additive noise in acceleration
    dW = np.sqrt(dt) * rng.standard_normal(n - 1) if noise_sigma > 0 else np.zeros(n - 1)

    for i in range(n - 1):
        a_det = -(2.0 * gamma) * v[i] - (omega0 ** 2) * x[i] + (F[i] / m)
        x[i + 1] = x[i] + v[i] * dt
        v[i + 1] = v[i] + a_det * dt + (noise_sigma / m) * dW[i]

    return x, v

def estimate_transfer_fft(t, x, F, eps_frac=1e-3, window=True):
    """
    Estimate G(iw) ~ X(w) / F(w) using rFFT.
    eps_frac sets a floor: only trust frequencies where |F(w)| >= eps_frac * max|F(w)|.
    """
    dt = t[1] - t[0]
    n = len(t)

    if window:
        w = np.hanning(n)
        xw = x * w
        Fw = F * w
        # Simple amplitude correction for window power (optional-ish); ratio cancels a lot anyway.
    else:
        xw, Fw = x, F

    X = np.fft.rfft(xw)
    U = np.fft.rfft(Fw)
    freqs = np.fft.rfftfreq(n, d=dt)          # Hz
    omega = 2.0 * np.pi * freqs               # rad/s

    magU = np.abs(U)
    floor = eps_frac * np.max(magU)
    valid = magU >= floor

    G = np.full_like(X, np.nan + 1j*np.nan, dtype=np.complex128)
    G[valid] = X[valid] / U[valid]

    return freqs, omega, G, valid

def analytic_G(omega, m, omega0, gamma):
    # G_xF(iw) = 1 / (m(omega0^2 - omega^2 + i 2 gamma omega))
    return 1.0 / (m * ((omega0**2 - omega**2) + 1j * (2.0 * gamma * omega)))

# %%


# --- Parameters ---
m = 1.0
omega0 = 2*np.pi*10      # 10 Hz natural frequency
gamma = 3.0              # 1/s
dt = 1e-4
t_max = 5.0
t = np.arange(0.0, t_max, dt)

# Pulse: choose sigma to set your effective bandwidth (~ 1/(2*pi*sigma) scale in Hz)
t0 = 0.2
sigma = 0.005            # 5 ms pulse width
F = gaussian_pulse(t, t0=t0, sigma=sigma, area=1.0)

# w = np.hanning(len(t))
# xw = x * w
# Fw = F * w

# Simulate response
x, v = simulate_force_to_displacement(t, F, m=m, omega0=omega0, gamma=gamma, noise_sigma=.05)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
# ax[0].plot(t, F)
# ax[0].plot(t, F * w)
ax[0].plot(t, w)
ax[1].plot(t, x)
ax[1].plot(t, .1*v)



# %%


# Estimate transfer function
freqs, omega, Ghat, valid = estimate_transfer_fft(t, x, F, eps_frac=1e-3, window=True)

# Compare to analytic transfer function (sanity check)
Gtrue = analytic_G(omega, m=m, omega0=omega0, gamma=gamma)

# --- Plot ---
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
ax[0].plot(freqs[valid], np.abs(Ghat[valid]), label="|Ghat| from pulse/FFT")
ax[0].plot(freqs[valid], np.abs(Gtrue[valid]), "--", label="|G| analytic")
ax[0].set_ylabel("Magnitude |X/F|")
ax[0].legend()

ax[1].plot(freqs[valid], np.unwrap(np.angle(Ghat[valid])), label="∠Ghat")
ax[1].plot(freqs[valid], np.unwrap(np.angle(Gtrue[valid])), "--", label="∠G analytic")
ax[1].set_ylabel("Phase (rad)")
ax[1].set_xlabel("Frequency (Hz)")
ax[1].legend()

plt.tight_layout()
plt.show()

# %%
import numpy as np

def averaged_transfer_function(
    t,
    x_ensemble,
    F_ensemble,
    window=True,
    eps_frac=1e-12
):
    """
    Estimate G(iw) = S_xF / S_FF by ensemble averaging.

    Parameters
    ----------
    t : (N,) array
        Time array (uniform sampling)
    x_ensemble : (M, N) array
        Displacement responses, M realizations
    F_ensemble : (M, N) array
        Input forces, same M realizations
    window : bool
        Apply Hann window before FFT
    eps_frac : float
        Regularization floor for S_FF to avoid division by ~0

    Returns
    -------
    freqs : (Nf,) array
        Frequencies (Hz)
    omega : (Nf,) array
        Angular frequencies (rad/s)
    Ghat : (Nf,) complex array
        Estimated transfer function
    valid : (Nf,) bool array
        Frequencies where estimate is trustworthy
    """
    dt = t[1] - t[0]
    M, N = x_ensemble.shape

    if window:
        w = np.hanning(N)
        # Window power normalization (important for spectra)
        Wnorm = np.mean(w**2)
    else:
        w = np.ones(N)
        Wnorm = 1.0

    S_xF = None
    S_FF = None

    for m in range(M):
        xw = x_ensemble[m] * w
        Fw = F_ensemble[m] * w

        X = np.fft.rfft(xw)
        Fhat = np.fft.rfft(Fw)

        if S_xF is None:
            S_xF = X * np.conj(Fhat)
            S_FF = Fhat * np.conj(Fhat)
        else:
            S_xF += X * np.conj(Fhat)
            S_FF += Fhat * np.conj(Fhat)

    # Ensemble average
    S_xF /= M * Wnorm
    S_FF /= M * Wnorm

    freqs = np.fft.rfftfreq(N, d=dt)
    omega = 2.0 * np.pi * freqs

    # Regularization + validity mask
    floor = eps_frac * np.max(S_FF.real)
    valid = S_FF.real > floor

    Ghat = np.full_like(S_xF, np.nan + 1j*np.nan)
    Ghat[valid] = S_xF[valid] / S_FF[valid]

    return freqs, omega, Ghat, valid

# %%
M = 200  # number of noise realizations

x_ens = np.zeros((M, len(t)))
F_ens = np.zeros((M, len(t)))

for m in range(M):
    F = gaussian_pulse(t, t0=0.2, sigma=0.005, area=1.0)
    x, _ = simulate_force_to_displacement(
        t, F,
        m=1.0,
        omega0=omega0,
        gamma=gamma,
        noise_sigma=.05,   # noisy bath
        seed=m
    )
    x_ens[m] = x
    F_ens[m] = F

freqs, omega, Gavg, valid = averaged_transfer_function(
    t, x_ens, F_ens, window=True
)

# %%
plt.plot(freqs, np.abs(Gavg))
plt.xlim(0, 25)
plt.ylim(-2e-3,2e-2)
# %%
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
ax[0].plot(freqs[valid == True], np.abs(Gavg[valid == True]))
ax[0].set_xlim(0, 25)
ax[0].set_ylim(0, 4e-2)

ax[1].plot(freqs[valid == True], np.angle(Gavg[valid == True]))
ax[1].set_xlim(0, 25)
# ax[1].ylim(-2e-2,2e-2)

# %%
freqs[valid == True]
# %%
