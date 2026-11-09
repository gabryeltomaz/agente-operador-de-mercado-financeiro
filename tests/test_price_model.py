import dataclasses

import numpy as np
import pytest

from udicoin.market.diagnostico import metricas_ancora
from udicoin.market.price_model import (
    LATERAL,
    ConfigMercado,
    Oculto,
    ProcessoDePreco,
    Tick,
    distribuicao_estacionaria,
    simular,
)


def test_mesma_semente_mesma_trajetoria(ln_sintetico):
    a = simular(ln_sintetico[:500], semente=3)
    b = simular(ln_sintetico[:500], semente=3)
    c = simular(ln_sintetico[:500], semente=4)
    assert np.array_equal(a["close"], b["close"])
    assert not np.array_equal(a["close"], c["close"])


def test_ancora_reproduz_os_retornos_do_dolar(ln_sintetico):
    r = simular(ln_sintetico[:1000], semente=1)
    np.testing.assert_allclose(np.diff(np.log(r["ancora"])), np.diff(ln_sintetico[:1000]), atol=1e-12)
    assert r["ancora"][0] == pytest.approx(100.0)  # reescalada para começar em p0


def test_velas_sao_consistentes(ln_sintetico):
    r = simular(ln_sintetico[:2000], semente=2)
    assert r["abertura"][0] == pytest.approx(100.0)
    np.testing.assert_allclose(r["abertura"][1:], r["close"][:-1])
    assert (r["maxima"] >= np.maximum(r["abertura"], r["close"])).all()
    assert (r["minima"] <= np.minimum(r["abertura"], r["close"])).all()
    assert (r["minima"] > 0).all() and (r["volume"] > 0).all()
    assert np.isfinite(r["close"]).all()


def test_variancia_do_desvio_bate_com_a_teoria():
    """Sem regimes e com âncora parada, d é um AR(1): Var(d) = σ² / (1 - (1-k)²)."""
    cfg = ConfigMercado(mu=(0.0, 0.0, 0.0), sigma=(0.004, 0.004, 0.004))
    r = simular(np.zeros(100_000), cfg, semente=7)
    desvio = np.log(r["close"] / r["ancora"])
    teoria = 0.004**2 / (1 - (1 - cfg.k) ** 2)
    assert np.var(desvio[1000:]) == pytest.approx(teoria, rel=0.10)
    assert abs(desvio.mean()) < 0.003


def test_o_elastico_puxa_de_volta():
    """Com k maior o desvio fica menor (e com k = 0 ele vaga sem voltar)."""
    cfg = lambda k: ConfigMercado(k=k, mu=(0.0, 0.0, 0.0), sigma=(0.004, 0.004, 0.004))
    dp = lambda k: np.std(np.log((s := simular(np.zeros(40_000), cfg(k), 1))["close"] / s["ancora"]))
    assert dp(0.1) < dp(0.02) < dp(0.005)


def test_cadeia_de_regimes_chega_na_distribuicao_estacionaria():
    cfg = ConfigMercado()
    r = simular(np.zeros(100_000), cfg, semente=5)
    freq = np.bincount(r["regime"], minlength=3) / len(r["regime"])
    np.testing.assert_allclose(freq, distribuicao_estacionaria(np.array(cfg.transicao)), atol=0.04)
    assert distribuicao_estacionaria(np.array(cfg.transicao)).sum() == pytest.approx(1.0)


def test_impacto_do_fluxo_e_exato_e_decai():
    """O fluxo de hoje empurra o desvio por λ·fluxo/profundidade, e o elástico o devolve."""
    ln = np.zeros(20)
    cfg = ConfigMercado()

    def rodar(fluxo):
        p = ProcessoDePreco(ln, cfg, np.random.default_rng(11), regime_inicial=LATERAL)
        return np.array([p.step(fluxo if t == 5 else 0.0)[0].fechamento for t in range(20)])

    base, empurrado = rodar(0.0), rodar(2e6)
    delta = cfg.impacto_lambda * 2e6 / cfg.impacto_profundidade
    np.testing.assert_allclose(empurrado[:5], base[:5])
    esperado = np.exp(delta * (1 - cfg.k) ** np.arange(15))
    np.testing.assert_allclose(empurrado[5:] / base[5:], esperado, rtol=1e-12)


def test_venda_empurra_o_preco_para_baixo():
    ln = np.zeros(10)
    p1 = ProcessoDePreco(ln, rng=np.random.default_rng(2), regime_inicial=0)
    p2 = ProcessoDePreco(ln, rng=np.random.default_rng(2), regime_inicial=0)
    c1 = [p1.step(0.0)[0].fechamento for _ in range(5)][-1]
    c2 = [p2.step(-5e6)[0].fechamento if t == 4 else p2.step(0.0)[0].fechamento for t in range(5)][-1]
    assert c2 < c1


def test_tick_publico_nao_tem_o_valor_justo():
    p = ProcessoDePreco(np.zeros(5), rng=np.random.default_rng(0))
    tick, oculto = p.step()
    assert isinstance(tick, Tick) and isinstance(oculto, Oculto)
    escondidos = {"ancora", "desvio", "regime"}
    assert not (escondidos & {f.name for f in dataclasses.fields(Tick)})
    assert escondidos <= {f.name for f in dataclasses.fields(Oculto)}


def test_acabou_a_serie():
    p = ProcessoDePreco(np.zeros(3), rng=np.random.default_rng(0))
    for _ in range(3):
        p.step()
    with pytest.raises(IndexError):
        p.step()


def test_reset_recomeca_do_preco_inicial():
    p = ProcessoDePreco(np.zeros(10), rng=np.random.default_rng(0))
    for _ in range(5):
        p.step()
    p.reset("lateral")
    tick, oculto = p.step()
    assert tick.t == 0 and tick.abertura == 100.0 and oculto.regime == LATERAL


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 1.5},
        {"k": -0.1},
        {"p0": 0},
        {"impacto_profundidade": 0},
        {"sigma": (0.004, -0.1, 0.003)},
        {"transicao": ((0.5, 0.5, 0.5), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8))},
        {"mu": (0.0, 0.0)},
    ],
)
def test_config_invalida(kwargs):
    with pytest.raises(ValueError):
        ConfigMercado(**kwargs)


def test_entrada_invalida():
    with pytest.raises(ValueError):
        ProcessoDePreco(np.array([1.0]))
    with pytest.raises(ValueError):
        ProcessoDePreco(np.array([0.0, np.nan]))
    with pytest.raises(ValueError):
        simular(np.zeros(5), fluxos=np.zeros(3))


def test_configuracao_padrao_fica_perto_do_alvo(ln_sintetico):
    """Com k = 0,02 o desvio típico fica em poucos %, como medido no site (~3%)."""
    dps = [
        metricas_ancora((r := simular(ln_sintetico[:2000], semente=s))["close"], r["ancora"])["dp_pct"]
        for s in range(20)
    ]
    assert 2.2 <= np.mean(dps) <= 4.2
