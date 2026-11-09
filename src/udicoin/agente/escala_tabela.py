"""A tabela Q escala? Tabela pequena (27 estados) x tabela GRANDE (2187 estados) x DQN.

    python -m udicoin.agente.escala_tabela --nivel 5

A tabela grande usa as 7 features, cada uma em 3 faixas (3^7 = 2187 estados, 6561 notas). É a mesma informação que
o DQN lê, só que "fatiada". Se a tabela grande aprender mais devagar (ou pior), é o argumento da maldição da dimensão.
Treina cada tabela com poucos e com muitos episódios; o DQN vem dos pesos já treinados em pesos/.
Mede em 200 episódios de validação novos (média ± erro padrão).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from udicoin.agente.avaliar_dqn import episodios
from udicoin.agente.dqn import AgenteDQN
from udicoin.agente.q_learning import AgenteQ, discretizar, treinar
from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv
from udicoin.market.price_model import ConfigMercado

RAIZ = Path(__file__).resolve().parents[3]
N_GRANDE = 3 ** 7


def discretizar_grande(obs) -> int:
    """Cada uma das 7 features vira 0, 1 ou 2 (baixa, média, alta) e juntamos tudo num índice 0..2186."""
    lim = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5])  # vol_20d e exposição têm limiares próprios abaixo
    s = 0
    for i, x in enumerate(obs):
        if i == 4:      # volatilidade (0..3): baixa < 0,8 < média < 1,2 < alta
            f = 0 if x < 0.8 else (2 if x > 1.2 else 1)
        elif i == 5:    # exposição (0..1)
            f = 0 if x < 0.25 else (1 if x < 0.75 else 2)
        else:           # as demais estão em desvios-padrão (centradas em 0)
            f = 0 if x < -lim[i] else (2 if x > lim[i] else 1)
        s = s * 3 + f
    return s


def main(nivel=5.0, et=None, ev=None, pesos=None, n=200, episodios_lista=(300, 3000), sementes=(0, 1, 2)):
    cfg = ConfigMercado(mu=tuple(x * nivel for x in ConfigMercado().mu))
    et = et or UdiCoinEnv.do_csv(ConfigAmbiente(divisao="treino", mercado=cfg))
    ev = ev or UdiCoinEnv.do_csv(ConfigAmbiente(divisao="validacao", mercado=cfg))
    seeds = range(5000, 5000 + n)
    print(f"nível {nivel:g}x, {n} episódios de validação novos (média ± erro padrão; média de {len(sementes)} sementes)\n")
    print(f"{'método':>28} | {'treino (episódios)':>18} | {'retorno%':>16} | {'exp':>5}")
    for nome, disc, n_est in (("tabela pequena (27)", discretizar, 27), ("tabela grande (2187)", discretizar_grande, N_GRANDE)):
        for ep in episodios_lista:
            rs, es = [], []
            for sm in sementes:
                ag = AgenteQ(alfa=0.02, gama=0.99, semente=sm, n_estados=n_est)
                treinar(et, ag, ep, semente=sm, disc=disc)
                r, e = episodios(ev, lambda o: ag.agir(disc(o), 0.0), seeds)
                rs.append(r); es.append(e.mean())
            todos = np.concatenate(rs)
            visit = ""
            print(f"{nome:>28} | {ep:18d} | {todos.mean():+7.2f} ± {todos.std() / np.sqrt(len(todos)):4.2f} | {np.mean(es):5.2f}", flush=True)
    pesos = Path(pesos) if pesos else RAIZ / "pesos"
    arq = next((c for c in (pesos / f"dqn_{nivel:g}x_s0_final.pt", pesos / f"dqn_{nivel}x_s0_final.pt") if c.exists()), None)
    if arq:
        ag = AgenteDQN(); ag.rede.load_state_dict(torch.load(arq))
        r, e = episodios(ev, lambda o: ag.agir(o, 0.0), seeds)
        print(f"{'DQN (7 features contínuas)':>28} | {'3000 (já treinado)':>18} | {r.mean():+7.2f} ± {r.std() / np.sqrt(len(r)):4.2f} | {e.mean():5.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nivel", type=float, default=5.0)
    a = p.parse_args()
    main(a.nivel)
