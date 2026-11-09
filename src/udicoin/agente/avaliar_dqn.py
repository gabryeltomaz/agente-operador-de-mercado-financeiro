"""Reavalia as redes salvas em episódios NOVOS e compara com a tabela Q e com comprar e segurar.

    python -m udicoin.agente.avaliar_dqn --nivel 5

Usa 200 episódios de validação com sementes que o treino nunca usou (5000 a 5199). Mostra média ± erro padrão
(desvio/√n): se dois métodos diferem por menos que ~2 erros padrão, não dá para dizer que um é melhor.
Procura pesos/dqn_<nivel>x_s0_melhor.pt e ..._final.pt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from udicoin.agente.dqn import AgenteDQN
from udicoin.agente.experimentos_q import _bh
from udicoin.agente.q_learning import AgenteQ, discretizar, treinar
from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv
from udicoin.market.price_model import ConfigMercado

RAIZ = Path(__file__).resolve().parents[3]


def episodios(env, escolher, seeds):
    """Retorno % de cada episódio e exposição média de cada um."""
    rets, exps = [], []
    for sd in seeds:
        obs, info = env.reset(seed=sd)
        eq0, fim, e = info["patrimonio"], False, []
        while not fim:
            obs, r, _, fim, info = env.step(escolher(obs))
            e.append(info["exposicao"])
        rets.append(100 * (info["patrimonio"] / eq0 - 1)); exps.append(np.mean(e))
    return np.array(rets), np.array(exps)


def main(nivel=5.0, et=None, ev=None, pesos=None, n=200, semente_dqn=0):
    cfg = ConfigMercado(mu=tuple(x * nivel for x in ConfigMercado().mu))
    et = et or UdiCoinEnv.do_csv(ConfigAmbiente(divisao="treino", mercado=cfg))
    ev = ev or UdiCoinEnv.do_csv(ConfigAmbiente(divisao="validacao", mercado=cfg))
    pesos = Path(pesos) if pesos else RAIZ / "pesos"
    seeds = range(5000, 5000 + n)
    metodos = {}
    metodos["comprar e segurar"] = episodios(ev, lambda o: 2, seeds)
    metodos["ficar fora"] = episodios(ev, lambda o: 0, seeds)
    tab = AgenteQ(alfa=0.02, gama=0.99, semente=0)
    treinar(et, tab, 300, semente=0)
    metodos["tabela Q"] = episodios(ev, lambda o: tab.agir(discretizar(o), 0.0), seeds)
    for nome in ("melhor", "final"):
        # versões antigas do treino gravavam "5.0x"; as novas gravam "5x". Aceita os dois.
        candidatos = [pesos / f"dqn_{t}x_s{semente_dqn}_{nome}.pt" for t in (f"{nivel:g}", str(nivel))]
        arq = next((c for c in candidatos if c.exists()), None)
        if arq is None:
            print(f"(não achei {candidatos[0]} nem {candidatos[1]})"); continue
        ag = AgenteDQN()
        ag.rede.load_state_dict(torch.load(arq))
        metodos[f"DQN {nome}"] = episodios(ev, lambda o: ag.agir(o, 0.0), seeds)
    bh_ret = metodos["comprar e segurar"][0]
    print(f"{n} episódios de validação novos, nível {nivel:g}x\n")
    print(f"{'método':>18} | {'retorno% ± erro padrão':>22} | {'exp':>5} | {'ganha do b&h':>12}")
    for nome, (r, e) in metodos.items():
        print(f"{nome:>18} | {r.mean():+9.2f} ± {r.std() / np.sqrt(len(r)):4.2f}       | {e.mean():5.2f} | {np.mean(r > bh_ret) * 100:10.0f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nivel", type=float, default=5.0)
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    main(a.nivel, n=a.n)
