"""Kleines Netz von Hand in numpy: ein verstecktes tanh-Layer, ein Ausgabewert (Logit). Parameter als Liste [W1 (h, d), b1 (h,), W2 (h,), b2 ()] aus float64.

Zwei Netze der Demo benutzen es: das Zugnetz Z (d = 9, h = 16, 177 Parameter) und das Merkmalsnetz K (d = 7, h = 16, 145 Parameter, Quelle für die Folge-Stücke).
Rückwärtspass von Hand (Gradienten-Check gegen zentrale Differenzen in tests/test_mlp.py), Adam mit Normkappung, Modelldatei `zwl-mlp-1` (.npz ohne Pickle)."""

import math
from pathlib import Path

import numpy as np

import zwl_constants as C


def init_mlp(d: int, h: int, us) -> list:
    """Anfangsgewichte: W1 ~ N(0,1) / sqrt(d), W2 ~ N(0,1) * 0,3 / sqrt(h), Biase 0. `us` ist ein UStream (Normalzahlen in der Reihenfolge W1, W2)."""
    W1 = us.normal((h, d)) / math.sqrt(d)
    W2 = us.normal((h,)) * 0.3 / math.sqrt(h)
    return [W1, np.zeros(h), W2, np.zeros(())]


def forward(params, X: np.ndarray) -> np.ndarray:
    """Logits z = W2 . tanh(W1 x + b1) + b2 für Zeilen X (N, d) -> (N,)."""
    W1, b1, W2, b2 = params
    return np.tanh(X @ W1.T + b1) @ W2 + b2


def n_params(params) -> int:
    return int(sum(p.size for p in params))


def copy_params(params) -> list:
    return [np.array(p, dtype=np.float64, copy=True) for p in params]


def softplus(z: np.ndarray) -> np.ndarray:
    """log(1 + e^z), stabil."""
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss_grad(params, X: np.ndarray, y: np.ndarray):
    """Logistischer Verlust (Mittel über die Zeilen) softplus(z) - y z und sein Gradient von Hand. X (N, d), y (N,) in {0, 1}. Gibt (Verlust, [gW1, gb1, gW2, gb2])."""
    W1, b1, W2, b2 = params
    H = np.tanh(X @ W1.T + b1)
    z = H @ W2 + b2
    N = len(y)
    verlust = float((softplus(z) - y * z).mean())
    dz = (sigmoid(z) - y) / N
    dpre = dz[:, None] * W2[None, :] * (1.0 - H ** 2)
    return verlust, [dpre.T @ X, dpre.sum(0), H.T @ dz, np.array(dz.sum())]


class Adam:
    """Adam mit optionaler Kappung der Gradientennorm; verändert die Parameter an Ort und Stelle."""

    def __init__(self, params, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, clip: float | None = None) -> float:
        """Ein Schritt; gibt die Norm des Gradienten vor der Kappung zurück."""
        norm = math.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if clip is not None and norm > clip:
            grads = [g * (clip / norm) for g in grads]
        self.t += 1
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return norm


# ---------------------------------------------------------------------------------------------------------------- Modelldatei
def save_model(path, params, meta: dict) -> None:
    """Modell als .npz ohne Pickle: Schlüssel W1, b1, W2, b2 und Skalare (Format, Art, Gewichts-Zufalls-Seed, Schritte, Lernrate, Trainingstouren oder -zeilen, Merkmalsversion, weitere Angaben)."""
    skalare = {k: np.array(v) for k, v in meta.items()}
    np.savez_compressed(str(path), W1=params[0], b1=params[1], W2=params[2], b2=params[3], format=np.array(C.MODELL_FORMAT), **skalare)


def load_model(path):
    """(Parameter, Angaben) aus einer Modelldatei; prüft Format, Formen und Parameterzahl, sonst ValueError."""
    with np.load(str(path), allow_pickle=False) as z:
        felder = {k: z[k] for k in z.files}
    if str(felder.get("format", "")) != C.MODELL_FORMAT:
        raise ValueError(f"unbekanntes Modellformat: {felder.get('format')}")
    art = str(felder.get("kind", ""))
    d = {"Z": C.Z_MERKMALE, "K": C.K_MERKMALE}.get(art)
    if d is None:
        raise ValueError(f"unbekannte Modellart: {art!r}")
    soll = [(C.H, d), (C.H,), (C.H,), ()]
    params = [np.array(felder[k], dtype=np.float64) for k in ("W1", "b1", "W2", "b2")]
    if [p.shape for p in params] != soll:
        raise ValueError(f"falsche Formen {[p.shape for p in params]}, erwartet {soll}")
    if n_params(params) != {"Z": C.Z_PARAMS, "K": C.K_PARAMS}[art]:
        raise ValueError("falsche Parameterzahl")
    if not all(np.isfinite(p).all() for p in params):
        raise ValueError("Gewichte nicht endlich")
    meta = {k: (v.item() if v.ndim == 0 else v) for k, v in felder.items() if k not in ("W1", "b1", "W2", "b2")}
    return params, meta


def model_path(art: str, seed: int, root: Path | None = None) -> Path:
    """Pfad des vorgerechneten Modells: zwl_modelle/zugnetz_s<seed>.npz (Z) bzw. featnet_s<seed>.npz (K)."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    return root / C.MODELL_ORDNER / f"{'zugnetz' if art == 'Z' else 'featnet'}_s{int(seed)}.npz"
