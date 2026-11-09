"""Treina o Q-learning no treino e compara com as referências na validação:  python -m udicoin.agente.treinar_q"""
from __future__ import annotations

import numpy as np

from udicoin.agente.q_learning import AgenteQ, avaliar, descrever_estado, discretizar, treinar
from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv

NOMES_ACAO = ("fora", "meio", "TUDO")


def main(env_treino=None, env_val=None, episodios: int = 300) -> AgenteQ:
    env_treino = env_treino or UdiCoinEnv.do_csv(ConfigAmbiente(divisao="treino"))
    env_val = env_val or UdiCoinEnv.do_csv(ConfigAmbiente(divisao="validacao"))
    agente = AgenteQ()
    ret = treinar(env_treino, agente, episodios)
    print(f"Treino: retorno médio (log) dos 1os 50 episódios = {ret[:50].mean():+.3f}, dos últimos 50 = {ret[-50:].mean():+.3f}")

    seeds = range(1000, 1050)  # episódios de validação que o agente nunca viu
    r_agente = avaliar(env_val, lambda o: agente.agir(discretizar(o), 0.0), seeds)
    r_tudo = avaliar(env_val, lambda o: 2, seeds)
    r_fora = avaliar(env_val, lambda o: 0, seeds)
    print("\nValidação (50 episódios novos), retorno % médio / pior / melhor:")
    for nome, r in (("agente Q", r_agente), ("comprar e segurar", r_tudo), ("ficar fora", r_fora)):
        print(f"  {nome:18}{r.mean():7.2f}% {r.min():8.1f}% {r.max():8.1f}%")

    print("\nPolítica aprendida (estado -> melhor ação):")
    visitados = agente.Q.any(axis=1)
    for s, a in enumerate(agente.politica()):
        if visitados[s]:
            print(f"  {descrever_estado(s)}  ->  {NOMES_ACAO[a]}")
    return agente


if __name__ == "__main__":
    main()
