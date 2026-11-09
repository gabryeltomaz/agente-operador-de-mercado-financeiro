import numpy as np
import pytest

from udicoin.env.features import LOOKBACK, NOMES, calcular_features


def test_formato_e_limites():
    c = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 60)))
    f = calcular_features(c, 0.5, 0.01)
    assert f.shape == (len(NOMES),) and f.dtype == np.float32 and np.isfinite(f).all()
    assert (np.abs(f) <= 3).all()


def test_preco_parado_da_zero():
    f = calcular_features(np.full(30, 100.0), 0.0, 0.0)
    assert (f[:5] == 0).all()


def test_so_usa_os_ultimos_fechamentos():
    c = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 80)))
    np.testing.assert_array_equal(calcular_features(c, 0.2, 0), calcular_features(c[-(LOOKBACK + 1):], 0.2, 0))


def test_tendencia_de_alta_tem_sinal_positivo():
    f = calcular_features(100 * 1.01 ** np.arange(30), 0, 0)
    assert f[2] > 0 and f[3] > 0


def test_desvio_opcional_e_historico_curto():
    assert len(calcular_features(np.full(30, 100.0), 0, 0, desvio=0.02)) == len(NOMES) + 1
    with pytest.raises(ValueError):
        calcular_features(np.full(10, 100.0), 0, 0)
