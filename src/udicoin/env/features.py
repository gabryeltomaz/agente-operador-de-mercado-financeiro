"""Passo 0.4: o que o agente enxerga (só o público: preço passado e a própria conta).

Todas as features usam escalas FIXAS (nada é calculado com dados do futuro nem do episódio inteiro)
e são cortadas em ±3, para a rede neural receber números de tamanho parecido.
"""
from __future__ import annotations

import numpy as np

LOOKBACK = 20  # precisa de LOOKBACK + 1 fechamentos
SIGMA_REF = 0.008  # volatilidade diária típica
NOMES = ("ret_1d", "ret_5d", "ret_20d", "razao_mm_5_20", "vol_20d", "exposicao", "lucro_posicao")
NOME_TRAPACA = "desvio_ancora"  # só o agente "trapaceiro" (teste de sanidade) vê isto


def _corta(x: float, lo: float = -3.0, hi: float = 3.0) -> float:
    return float(min(max(x, lo), hi))


def calcular_features(
    fechamentos: np.ndarray, exposicao: float, lucro_posicao: float, desvio: float | None = None
) -> np.ndarray:
    c = np.asarray(fechamentos, dtype=float)
    if len(c) < LOOKBACK + 1:
        raise ValueError(f"precisa de pelo menos {LOOKBACK + 1} fechamentos")
    c = c[-(LOOKBACK + 1) :]
    lnc = np.log(c)
    r = np.diff(lnc)
    f = [
        _corta((lnc[-1] - lnc[-2]) / SIGMA_REF),
        _corta((lnc[-1] - lnc[-6]) / (SIGMA_REF * 5**0.5)),
        _corta((lnc[-1] - lnc[0]) / (SIGMA_REF * LOOKBACK**0.5)),
        _corta((c[-5:].mean() / c.mean() - 1.0) / 0.01),
        _corta(r.std() / SIGMA_REF, 0.0, 3.0),
        _corta(exposicao, 0.0, 1.0),
        _corta(lucro_posicao / 0.05),
    ]
    if desvio is not None:
        f.append(_corta(desvio / 0.03))
    return np.array(f, dtype=np.float32)
