"""Medidas de quão solta ou colada está a udiCOIN em relação à âncora (o dólar)."""
from __future__ import annotations

import numpy as np


def metricas_ancora(close: np.ndarray, ancora: np.ndarray) -> dict[str, float]:
    """As mesmas medidas da função `check()` que você rodou no console do site."""
    close = np.asarray(close, dtype=float)
    ancora = np.asarray(ancora, dtype=float)
    desvio = np.log(close / ancora)
    r = np.diff(np.log(close))
    ra = np.diff(np.log(ancora))
    return {
        "dp_pct": float(desvio.std() * 100),
        "max_pct": float(np.abs(desvio).max() * 100),
        "corr": float(np.corrcoef(r, ra)[0, 1]),
        "vol_udi_pct": float(r.std() * 100),
        "vol_dolar_pct": float(ra.std() * 100),
    }
