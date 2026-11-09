"""Passo 4 (primeira varredura): como episódios, γ e α mudam o agente.  python -m udicoin.agente.experimentos_q

Para cada combinação treina com 2 sementes e mede, com ε = 0 em episódios novos:
  val%   retorno médio na validação      treino%  retorno médio em episódios do treino
  exp    exposição média (0 = nunca entrou, 1 = sempre comprado)
Referência de ouro: compare com "comprar e segurar" impresso no topo.
"""
from __future__ import annotations

import itertools

import numpy as np

from udicoin.agente.q_learning import AgenteQ, discretizar, treinar
from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv


def medir(env, agente, seeds):
    rets, exps = [], []
    for sd in seeds:
        obs, info = env.reset(seed=sd)
        eq0, fim, e = info["patrimonio"], False, []
        while not fim:
            obs, r, _, fim, info = env.step(agente.agir(discretizar(obs), 0.0))
            e.append(info["exposicao"])
        rets.append(100 * (info["patrimonio"] / eq0 - 1)); exps.append(np.mean(e))
    return np.mean(rets), np.mean(exps)


def main(env_treino=None, env_val=None, episodios=(300, 1500), gamas=(0.9, 0.99), alfas=(0.1, 0.02), sementes=(0, 1)):
    env_treino = env_treino or UdiCoinEnv.do_csv(ConfigAmbiente(divisao="treino"))
    env_val = env_val or UdiCoinEnv.do_csv(ConfigAmbiente(divisao="validacao"))
    seeds_val, seeds_tr = range(1000, 1030), range(2000, 2030)
    ref = np.mean([_bh(env_val, s) for s in seeds_val]); ref_tr = np.mean([_bh(env_treino, s) for s in seeds_tr])
    print(f"Comprar e segurar: validação {ref:+.2f}%   treino {ref_tr:+.2f}%\n")
    print(f"{'episódios':>9} {'γ':>5} {'α':>5} | {'val%':>7} {'treino%':>8} {'exp':>5}")
    for ep, g, a in itertools.product(episodios, gamas, alfas):
        v, t, x = [], [], []
        for sm in sementes:
            ag = AgenteQ(alfa=a, gama=g, semente=sm)
            treinar(env_treino, ag, ep, semente=sm)
            rv, ev = medir(env_val, ag, seeds_val); rt, _ = medir(env_treino, ag, seeds_tr)
            v.append(rv); t.append(rt); x.append(ev)
        print(f"{ep:9d} {g:5.2f} {a:5.2f} | {np.mean(v):+7.2f} {np.mean(t):+8.2f} {np.mean(x):5.2f}", flush=True)


def _bh(env, seed):
    obs, info = env.reset(seed=seed)
    eq0, fim = info["patrimonio"], False
    while not fim:
        obs, r, _, fim, info = env.step(2)
    return 100 * (info["patrimonio"] / eq0 - 1)


if __name__ == "__main__":
    main()
