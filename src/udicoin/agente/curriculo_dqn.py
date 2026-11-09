"""DQN nos mesmos degraus de dificuldade da tabela Q:  python -m udicoin.agente.curriculo_dqn

Compare com a tabela do curriculo da tabela Q (mesmas colunas). Se o DQN não for melhor, tudo bem: é resultado.
"""
from __future__ import annotations

import time

import numpy as np

from udicoin.agente.dqn import AgenteDQN, medir_dqn, treinar_dqn
from udicoin.agente.experimentos_q import _bh
from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv
from udicoin.market.price_model import ConfigMercado


def main(periodos=None, multiplicadores=(1, 3), episodios=300, sementes=(0, 1)):
    mk = (lambda cfg, div: UdiCoinEnv(periodos, ConfigAmbiente(divisao=div, mercado=cfg))) if periodos else (
        lambda cfg, div: UdiCoinEnv.do_csv(ConfigAmbiente(divisao=div, mercado=cfg)))
    base, seeds = ConfigMercado(), range(1000, 1030)
    print(f"{'tendência':>9} | {'bh%':>7} {'DQN%':>7} {'exp':>5} | {'metade do bh%':>13} | tempo")
    for m in multiplicadores:
        cfg = ConfigMercado(mu=tuple(x * m for x in base.mu))
        et, ev = mk(cfg, "treino"), mk(cfg, "validacao")
        bh, t0, res = np.mean([_bh(ev, s) for s in seeds]), time.time(), []
        for sm in sementes:
            ag = AgenteDQN(semente=sm)
            treinar_dqn(et, ag, episodios, semente=sm)
            res.append(medir_dqn(ev, ag, seeds))
        r, e = np.mean([x[0] for x in res]), np.mean([x[1] for x in res])
        print(f"{m:8d}x | {bh:+7.2f} {r:+7.2f} {e:5.2f} | {bh * e:+13.2f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
