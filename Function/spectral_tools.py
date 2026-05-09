import numpy as np

def s_transform(trace, dt, fmin, fmax, df, gamma=1.0):
    trace = np.asarray(trace)
    nt = trace.size
    t = np.arange(nt) * dt

    freqs = np.arange(fmin, fmax + df, df)
    S = np.zeros((len(freqs), nt), dtype=complex)

    for i, f in enumerate(freqs):
        if f == 0:
            continue

        sigma = gamma / abs(f)

        for tau in range(nt):
            window = np.exp(
                -((t - t[tau])**2) / (2 * sigma**2)
            )
            S[i, tau] = np.sum(
                trace * window * np.exp(-1j * 2 * np.pi * f * t)
            )

    return np.abs(S), freqs, t