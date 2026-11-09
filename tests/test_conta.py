import numpy as np
import pytest

from udicoin.market.conta import Conta, ConfigConta


def nova(**kw):
    return Conta(ConfigConta(**kw))


def test_comeca_so_com_caixa():
    c = nova()
    assert c.patrimonio(100) == 10_000 and c.exposicao(100) == 0 and c.lucro_posicao(100) == 0


@pytest.mark.parametrize("alvo", [0.5, 1.0])
def test_compra_chega_perto_do_alvo(alvo):
    c = nova()
    c.ajustar_para(alvo, 100.0)
    assert c.exposicao(100.0) == pytest.approx(alvo, abs=0.01)
    assert c.caixa >= 0


def test_identidade_contabil_em_sequencia_aleatoria():
    rng = np.random.default_rng(0)
    c = nova()
    preco = 100.0
    for _ in range(300):
        antes = c.patrimonio(preco)
        e = c.ajustar_para(float(rng.choice([0, 0.25, 0.5, 1.0])), preco)
        assert c.patrimonio(preco) == pytest.approx(antes - e.custo, rel=1e-12)
        assert c.caixa >= -1e-9 and c.unidades >= 0
        preco *= float(np.exp(rng.normal(0, 0.01)))


def test_ida_e_volta_so_perde_custos():
    c = nova()
    c.ajustar_para(1.0, 100.0)
    c.ajustar_para(0.0, 100.0)
    assert c.unidades == 0 and c.exposicao(100) == 0
    perda = 1 - c.patrimonio(100) / 10_000
    assert 0.002 < perda < 0.0035  # 2 x (0,1% taxa + 0,02% spread) + impacto pequeno


def test_banda_evita_operar_por_pouco():
    c = nova()
    c.ajustar_para(0.5, 100.0)
    assert c.ajustar_para(0.5, 100.5).tipo == "nada"


def test_alvo_zero_sempre_zera():
    c = nova(banda=0.5)
    c.ajustar_para(1.0, 100.0)
    c.ajustar_para(0.0, 100.0)
    assert c.unidades == 0


def test_compra_paga_mais_e_venda_recebe_menos_e_fluxo_tem_sinal():
    c = nova()
    e = c.ajustar_para(1.0, 100.0)
    assert e.preco_exec > 100 and e.fluxo > 0
    e = c.ajustar_para(0.0, 100.0)
    assert e.preco_exec < 100 and e.fluxo < 0


def test_ordem_maior_tem_mais_slippage():
    a, b = nova(capital_inicial=1e4), nova(capital_inicial=1e7)
    pa = a.ajustar_para(1.0, 100.0).preco_exec
    pb = b.ajustar_para(1.0, 100.0).preco_exec
    assert pb > pa


def test_lucro_da_posicao():
    c = nova()
    c.ajustar_para(1.0, 100.0)
    assert c.lucro_posicao(110.0) == pytest.approx(0.10, abs=0.002)


def test_alvo_invalido():
    for alvo in (-0.1, 1.1):
        with pytest.raises(ValueError):
            nova().ajustar_para(alvo, 100.0)
    with pytest.raises(ValueError):
        ConfigConta(capital_inicial=0)


def test_custo_do_caixa_so_afeta_o_dinheiro_parado():
    c = nova(custo_caixa=0.001)
    c.passar_dia()
    assert c.caixa == pytest.approx(10_000 * 0.999)
    c2 = nova(custo_caixa=0.001)
    c2.ajustar_para(1.0, 100.0)
    caixa_antes, unid = c2.caixa, c2.unidades
    c2.passar_dia()
    assert c2.unidades == unid and c2.caixa == pytest.approx(caixa_antes * 0.999)
    with pytest.raises(ValueError):
        ConfigConta(custo_caixa=1.0)
