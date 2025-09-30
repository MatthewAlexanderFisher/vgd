import jax, jax.numpy as jnp
import numpy as np


def hann_window(N: int) -> jnp.ndarray:
    n = jnp.arange(N)
    return 0.5 - 0.5 * jnp.cos(2*jnp.pi*n/(N-1))

def frame_signal(y: jnp.ndarray, frame_len: int, hop: int, pad_end: bool = True):
    """
    Return frames with hop spacing. If pad_end=True, zero-pad the tail so that
    the last frame is complete. Returns (frames, N_orig).
    """
    N = int(y.shape[0])
    if N <= frame_len:
        pad = frame_len - N if pad_end else 0
        y_pad = jnp.pad(y, (0, pad))
        return y_pad[None, :], N

    if pad_end:
        # ceil((N - frame_len)/hop) + 1 frames
        n_steps = (N - frame_len + hop - 1) // hop + 1
        total = hop * (n_steps - 1) + frame_len
        pad = max(0, total - N)
    else:
        n_steps = (N - frame_len) // hop + 1
        pad = 0

    y_pad = jnp.pad(y, (0, pad))
    idx = jnp.arange(frame_len)[None, :] + hop * jnp.arange(n_steps)[:, None]
    frames = y_pad[idx]  # (T, L)
    return frames, N  # keep original N for cropping later

def overlap_add(frames: jnp.ndarray, hop: int, window: jnp.ndarray, N_trim: int | None = None):
    """
    Overlap–add with window reweighting. If N_trim is provided, crop to that length.
    Assumes a COLA-compliant hop/window (e.g., Hann with hop=L/2).
    """
    T, L = frames.shape
    total = hop * (T - 1) + L
    y = jnp.zeros((total,))
    wsum = jnp.zeros((total,))
    for t in range(T):
        start = t * hop
        seg = frames[t] * window
        y = y.at[start:start+L].add(seg)
        wsum = wsum.at[start:start+L].add(window)

    if N_trim is not None:
        y = y[:N_trim]
        wsum = wsum[:N_trim]

    wsum = jnp.where(wsum > 1e-12, wsum, 1.0)
    return y / wsum

def snr_db(y_clean: np.ndarray, y_est: np.ndarray) -> float:
    num = np.sum(y_clean**2) + 1e-12
    den = np.sum((y_clean - y_est)**2) + 1e-12
    return float(10*np.log10(num/den))

def play_audio(y: np.ndarray, sr: int = 16000):
    try:
        from IPython.display import Audio, display
        display(Audio(y, rate=sr))
    except Exception as e:
        print("IPython Audio not available:", e)

def save_wav(path: str, y: np.ndarray, sr: int = 16000):
    import scipy.io.wavfile as wav
    y16 = np.clip(y, -1.0, 1.0)
    wav.write(path, sr, (y16*32767).astype(np.int16))


