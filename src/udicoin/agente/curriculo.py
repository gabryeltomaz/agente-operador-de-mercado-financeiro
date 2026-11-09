"""Degraus de dificuldade: o agente aprende quando o sinal é mais forte?  python -m udicoin.agente.curriculo

Mesmo agente (γ=0,99, α=0,02), mesmo ambiente; só muda a FORÇA das tendências escondidas (regimes de alta/baixa).
  1x = o padrão "realista"      3x, 5x = tendências mais fortes (mais fácil de aprender)
Se o agente aprende nos degraus fáceis e some no 1x, o código do agente está certo e o limite é o sinal do mercado.
Colunas: bh = comprar e segurar (validação) | agente = retorno na validação | exp = fração do tempo comprado
"""
from __future__ import annotations

import numpy as np

from udicoin.agente.experimentos_q import _bh, medir
from udicoin.agente.q_learning import AgenteQ, treinar
from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv
from udicoin.market.price_model import ConfigMercado


def main(periodos=None, multiplicadores=(1, 2, 3, 5), episodios=300, sementes=(0, 1)):
    mk = (lambda cfg, div: UdiCoinEnv(periodos, ConfigAmbiente(divisao=div, mercado=cfg))) if periodos else (
        lambda cfg, div: UdiCoinEnv.do_csv(ConfigAmbiente(divisao=div, mercado=cfg)))
    base = ConfigMercado()
    seeds = range(1000, 1030)
    print(f"{'tendência':>9} | {'bh%':>7} {'agente%':>8} {'exp':>5} | {'metade do bh%':>13}")
    for m in multiplicadores:
        cfg = ConfigMercado(mu=tuple(x * m for x in base.mu))
        et, ev = mk(cfg, "treino"), mk(cfg, "validacao")
        bh = np.mean([_bh(ev, s) for s in seeds])
        res = []
        for sm in sementes:
            ag = AgenteQ(alfa=0.02, gama=0.99, semente=sm)
            treinar(et, ag, episodios, semente=sm)
            res.append(medir(ev, ag, seeds))
        r, e = np.mean([x[0] for x in res]), np.mean([x[1] for x in res])
        print(f"{m:8d}x | {bh:+7.2f} {r:+8.2f} {e:5.2f} | {bh * e:+13.2f}", flush=True)


if __name__ == "__main__":
    main()
