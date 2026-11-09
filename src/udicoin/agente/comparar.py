"""Tabela Q x DQN, lado a lado, com a variação entre sementes:  python -m udicoin.agente.comparar

Para cada nível de tendência e cada método: treina com 3 sementes e mostra  média ± desvio entre sementes
(retorno % na validação, em 50 episódios novos) e a exposição média. Também imprime "comprar e segurar" e a régua
da mesma exposição. Leva ~25-30 min no PC (o DQN é a parte lenta). Imprime cada linha assim que termina.
"""
from __future__ import annotations

import time

import numpy as np

from udicoin.agente.dqn import AgenteDQN, medir_dqn, treinar_dqn
from udicoin.agente.experimentos_q import _bh, medir
from udicoin.agente.q_learning import AgenteQ, treinar
from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv
from udicoin.market.price_model import ConfigMercado


def _mm(x):
    return f"{np.mean(x):+7.2f} ± {np.std(x):4.1f}"


def main(periodos=None, multiplicadores=(3, 5), episodios_dqn=300, episodios_q=300, sementes=(0, 1, 2)):
    mk = (lambda cfg, div: UdiCoinEnv(periodos, ConfigAmbiente(divisao=div, mercado=cfg))) if periodos else (
        lambda cfg, div: UdiCoinEnv.do_csv(ConfigAmbiente(divisao=div, mercado=cfg)))
    base, seeds = ConfigMercado(), range(1000, 1050)
    print(f"{'tendência':>9} {'método':>9} | {'retorno% (média ± dp)':>22} | {'exp':>5} | {'régua':>7} | tempo")
    for m in multiplicadores:
        cfg = ConfigMercado(mu=tuple(x * m for x in base.mu))
        et, ev = mk(cfg, "treino"), mk(cfg, "validacao")
        bh = np.mean([_bh(ev, s) for s in seeds])
        print(f"{m:8d}x {'comprar+segurar':>9} | {bh:+22.2f} |")
        for nome, fazer in (("tabela Q", "q"), ("DQN", "dqn")):
            t0, r, e = time.time(), [], []
            for sm in sementes:
                if fazer == "q":
                    ag = AgenteQ(alfa=0.02, gama=0.99, semente=sm)
                    treinar(et, ag, episodios_q, semente=sm)
                    rv, ex = medir(ev, ag, seeds)
                else:
                    ag = AgenteDQN(semente=sm)
                    treinar_dqn(et, ag, episodios_dqn, semente=sm)
                    rv, ex = medir_dqn(ev, ag, seeds)
                r.append(rv); e.append(ex)
            print(f"{m:8d}x {nome:>9} | {_mm(r):>22} | {np.mean(e):5.2f} | {bh * np.mean(e):+7.2f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
