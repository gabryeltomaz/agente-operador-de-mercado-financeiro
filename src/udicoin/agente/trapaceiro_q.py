"""Teste de sanidade: o MESMO Q-learning, mas vendo o desvio escondido.  python -m udicoin.agente.trapaceiro_q

O "trapaceiro" enxerga d = ln(preço / valor justo). Isso NÃO vale como agente de verdade (no mundo real não
existe valor justo visível): serve só para medir o TETO do ambiente. Se nem ele ganha, o ambiente tem pouco sinal.
estado = (desvio em 5 faixas) x (exposição em 3 faixas) = 15 estados.
"""
from __future__ import annotations

import numpy as np

from udicoin.agente.experimentos_q import _bh, medir
from udicoin.agente.q_learning import AgenteQ, _faixa_exposicao, treinar
from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv

IDX_DESVIO = 7


def disc_trapaceiro(obs) -> int:
    z = obs[IDX_DESVIO]  # desvio em unidades de 3% (cortado em ±3)
    faixa = 0 if z < -1 else 1 if z < -0.3 else 2 if z <= 0.3 else 3 if z <= 1 else 4
    return faixa * 3 + _faixa_exposicao(obs[5])


class _Wrap:  # medir() chama discretizar; aqui trocamos pela do trapaceiro
    def __init__(self, ag): self.ag = ag
    def agir(self, s, eps): return self.ag.agir(s, eps)


def main(env_treino=None, env_val=None, episodios=300, sementes=(0, 1)):
    import udicoin.agente.experimentos_q as ex
    env_treino = env_treino or UdiCoinEnv.do_csv(ConfigAmbiente(divisao="treino", trapaceiro=True))
    env_val = env_val or UdiCoinEnv.do_csv(ConfigAmbiente(divisao="validacao", trapaceiro=True))
    ex.discretizar = disc_trapaceiro  # medir() passa a usar a discretização do trapaceiro
    seeds_val, seeds_tr = range(1000, 1030), range(2000, 2030)
    print(f"Comprar e segurar: validação {np.mean([_bh(env_val, s) for s in seeds_val]):+.2f}%   treino {np.mean([_bh(env_treino, s) for s in seeds_tr]):+.2f}%\n")
    print(f"{'γ':>5} {'α':>5} | {'val%':>7} {'treino%':>8} {'exp':>5}")
    for g, a in ((0.9, 0.1), (0.9, 0.02), (0.99, 0.02)):
        v, t, x = [], [], []
        for sm in sementes:
            ag = AgenteQ(alfa=a, gama=g, semente=sm, n_estados=15)
            treinar(env_treino, ag, episodios, semente=sm, disc=disc_trapaceiro)
            rv, ev = medir(env_val, ag, seeds_val); rt, _ = medir(env_treino, ag, seeds_tr)
            v.append(rv); t.append(rt); x.append(ev)
        print(f"{g:5.2f} {a:5.2f} | {np.mean(v):+7.2f} {np.mean(t):+8.2f} {np.mean(x):5.2f}", flush=True)


if __name__ == "__main__":
    main()
