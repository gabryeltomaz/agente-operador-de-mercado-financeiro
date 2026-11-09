"""Treino longo do DQN, com validação periódica e salvamento do melhor:

    python -m udicoin.agente.treino_dqn_longo --nivel 3 --episodios 3000

A cada `--avaliar-a-cada` episódios, mede a rede (ε = 0) na VALIDAÇÃO e grava a curva em CSV (bom para os slides).
Guarda em `pesos/` os pesos da melhor validação e os do final. Obs.: escolher o melhor pela validação já "gasta" um
pouco dela; o teste (2023+) só deve ser usado UMA vez, no fim, com a rede escolhida.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from udicoin.agente.dqn import AgenteDQN, medir_dqn
from udicoin.agente.experimentos_q import _bh
from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv
from udicoin.market.price_model import ConfigMercado

RAIZ = Path(__file__).resolve().parents[3]


def treinar_longo(et, ev, nivel=3, episodios=3000, avaliar_a_cada=250, semente=0, treinar_a_cada=2, saida=None, lr=1e-3):
    saida = Path(saida) if saida else RAIZ / "pesos"
    saida.mkdir(parents=True, exist_ok=True)
    etiqueta = f"dqn_{nivel:g}x_s{semente}"
    seeds_val = range(1000, 1030)
    bh = float(np.mean([_bh(ev, s) for s in seeds_val]))
    ag = AgenteDQN(semente=semente, treinar_a_cada=treinar_a_cada, capacidade=100_000, lr=lr)
    linhas, recentes, melhor, t0 = [], [], -1e9, time.time()
    print(f"nível {nivel}x | comprar e segurar na validação: {bh:+.2f}%")
    print(f"{'ep':>6} {'ε':>5} | {'treino(log, últ.50)':>19} | {'val%':>7} {'exp':>5} | {'régua':>7} | tempo")
    for ep in range(1, episodios + 1):
        eps = max(0.05, 1.0 - 0.95 * (ep - 1) / (0.7 * episodios))
        obs, info = et.reset(seed=semente * 100_000 + ep)
        fim, soma = False, 0.0
        while not fim:
            a = ag.agir(obs, eps)
            obs2, r, _, fim, info = et.step(a)
            ag.aprender(obs, a, r, obs2)
            obs, soma = obs2, soma + r
        recentes = (recentes + [soma])[-50:]
        if ep % avaliar_a_cada == 0 or ep == episodios:
            val, exp = medir_dqn(ev, ag, seeds_val)
            linhas.append(dict(episodio=ep, epsilon=round(eps, 3), treino_log=round(float(np.mean(recentes)), 4),
                               val_pct=round(val, 2), exposicao=round(exp, 3), comprar_segurar=round(bh, 2)))
            print(f"{ep:6d} {eps:5.2f} | {np.mean(recentes):+19.3f} | {val:+7.2f} {exp:5.2f} | {bh * exp:+7.2f} | {time.time() - t0:.0f}s", flush=True)
            if val > melhor:
                melhor = val
                torch.save(ag.rede.state_dict(), saida / f"{etiqueta}_melhor.pt")
    torch.save(ag.rede.state_dict(), saida / f"{etiqueta}_final.pt")
    with open(saida / f"{etiqueta}_curva.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)
    print(f"\nMelhor validação: {melhor:+.2f}%. Pesos e curva em {saida}")
    return ag, linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nivel", type=float, default=3, help="multiplicador da força das tendências (1 = realista)")
    p.add_argument("--episodios", type=int, default=3000)
    p.add_argument("--avaliar-a-cada", type=int, default=250)
    p.add_argument("--semente", type=int, default=0)
    a = p.parse_args()
    base = ConfigMercado()
    cfg = ConfigMercado(mu=tuple(x * a.nivel for x in base.mu))
    et = UdiCoinEnv.do_csv(ConfigAmbiente(divisao="treino", mercado=cfg))
    ev = UdiCoinEnv.do_csv(ConfigAmbiente(divisao="validacao", mercado=cfg))
    treinar_longo(et, ev, a.nivel, a.episodios, a.avaliar_a_cada, a.semente)


if __name__ == "__main__":
    main()
