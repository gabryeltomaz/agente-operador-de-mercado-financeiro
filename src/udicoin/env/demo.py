"""Roda 3 políticas triviais no ambiente: python -m udicoin.env.demo"""
from __future__ import annotations

import numpy as np

from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv


def rodar(env: UdiCoinEnv, politica, semente: int) -> dict:
    obs, info = env.reset(seed=semente)
    p0, eq0, soma, fim = info["fechamento"], info["patrimonio"], 0.0, False
    while not fim:
        obs, r, _, fim, info = env.step(politica(obs))
        soma += r
    return {"retorno_pct": 100 * (info["patrimonio"] / eq0 - 1), "soma_r": soma, "preco_pct": 100 * (info["fechamento"] / p0 - 1)}


def main() -> None:
    env = UdiCoinEnv.do_csv(ConfigAmbiente(divisao="treino"))
    rng = np.random.default_rng(0)
    politicas = {
        "zerada (sempre 0)": lambda o: 0,
        "comprada (sempre 100%)": lambda o: 2,
        "aleatória": lambda o: int(rng.integers(env.n_acoes)),
    }
    print(f"{'política':26}{'retorno médio':>15}{'pior':>9}{'melhor':>9}   (20 episódios de 252 dias)")
    for nome, pol in politicas.items():
        ret = np.array([rodar(env, pol, s)["retorno_pct"] for s in range(20)])
        print(f"{nome:26}{ret.mean():14.2f}%{ret.min():8.1f}%{ret.max():8.1f}%")


if __name__ == "__main__":
    main()
