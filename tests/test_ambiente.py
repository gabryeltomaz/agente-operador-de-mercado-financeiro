import dataclasses
import math

import numpy as np
import pytest

from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv


def rodar(env, acao_fn, seed):
    obs, info = env.reset(seed=seed)
    eq0, rs, infos, fim = info["patrimonio"], [], [info], False
    while not fim:
        obs, r, term, fim, info = env.step(acao_fn(obs))
        rs.append(r); infos.append(info)
    return eq0, rs, infos


def test_episodio_tem_252_passos_e_trunca(periodos_sint):
    env = UdiCoinEnv(periodos_sint)
    _, rs, infos = rodar(env, lambda o: 1, 0)
    assert len(rs) == 252 and infos[-1]["passo"] == 252
    with pytest.raises(RuntimeError):
        env.step(0)


def test_recompensas_somam_o_log_do_patrimonio(periodos_sint):
    env = UdiCoinEnv(periodos_sint)
    rng = np.random.default_rng(3)
    eq0, rs, infos = rodar(env, lambda o: int(rng.integers(3)), 5)
    assert sum(rs) == pytest.approx(math.log(infos[-1]["patrimonio"] / eq0), abs=1e-9)


def test_sempre_zerada_rende_zero(periodos_sint):
    _, rs, infos = rodar(UdiCoinEnv(periodos_sint), lambda o: 0, 1)
    assert all(r == 0 for r in rs) and infos[-1]["patrimonio"] == 10_000


def test_mesma_semente_mesmo_episodio(periodos_sint):
    env = UdiCoinEnv(periodos_sint)
    a = rodar(env, lambda o: 2, 7)[2][-1]["patrimonio"]
    b = rodar(env, lambda o: 2, 7)[2][-1]["patrimonio"]
    c = rodar(env, lambda o: 2, 8)[2][-1]["patrimonio"]
    assert a == b and a != c


def test_observacao_nao_vaza_o_oculto(periodos_sint):
    env = UdiCoinEnv(periodos_sint)
    obs, info = env.reset(seed=0)
    assert env.dim_obs == len(obs) == 7 and "desvio_ancora" not in env.nomes_obs
    # mudar o oculto não muda a observação
    antes = env._obs().copy()
    env._oculto = dataclasses.replace(env._oculto, desvio=0.5, regime=1)
    np.testing.assert_array_equal(antes, env._obs())


def test_trapaceiro_ve_o_desvio(periodos_sint):
    env = UdiCoinEnv(periodos_sint, ConfigAmbiente(trapaceiro=True))
    obs, info = env.reset(seed=0)
    assert env.nomes_obs[-1] == "desvio_ancora" and len(obs) == 8
    assert obs[-1] == pytest.approx(np.clip(info["oculto"].desvio / 0.03, -3, 3), abs=1e-6)


def test_acao_invalida(periodos_sint):
    env = UdiCoinEnv(periodos_sint)
    env.reset(seed=0)
    for a in (-1, 3, 1.0, "x", True):
        with pytest.raises(ValueError):
            env.step(a)
    with pytest.raises(RuntimeError):
        UdiCoinEnv(periodos_sint).step(0)


def test_ordem_de_hoje_so_mexe_no_preco_de_amanha(periodos_sint):
    env = UdiCoinEnv(periodos_sint)
    _, info = env.reset(seed=2)
    preco_hoje = info["fechamento"]
    _, _, _, _, info = env.step(2)
    assert info["execucao"].tipo == "compra" and info["execucao"].fluxo > 0
    assert env.historico_fechamentos[-2] == preco_hoje  # fechamento de hoje não foi alterado


def test_divisoes_respeitadas(periodos_sint):
    for nome in ("treino", "validacao", "teste"):
        env = UdiCoinEnv(periodos_sint, ConfigAmbiente(divisao=nome))
        _, info = env.reset(seed=0)
        d = periodos_sint[nome].datas
        assert d.min() <= info["data_real"] <= d.max()
    with pytest.raises(KeyError):
        UdiCoinEnv(periodos_sint, ConfigAmbiente(divisao="producao"))


def test_config_invalida():
    with pytest.raises(ValueError):
        ConfigAmbiente(aquecimento=5)
    with pytest.raises(ValueError):
        ConfigAmbiente(niveis=(0, 2))


def test_custo_do_caixa_faz_ficar_fora_render_negativo(periodos_sint):
    from udicoin.market.conta import ConfigConta
    env = UdiCoinEnv(periodos_sint, ConfigAmbiente(conta=ConfigConta(custo_caixa=0.0002)))
    _, rs, infos = rodar(env, lambda o: 0, 1)
    assert all(r < 0 for r in rs)
    assert infos[-1]["patrimonio"] == pytest.approx(10_000 * (1 - 0.0002) ** 252)
