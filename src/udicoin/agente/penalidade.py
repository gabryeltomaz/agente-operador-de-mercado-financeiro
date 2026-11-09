"""Penalidade por dinheiro parado: ela tira o agente do "ficar fora"?  python -m udicoin.agente.penalidade

`custo` = perda diária do caixa (0,0002 ≈ 5% ao ano). A régua passa a ser DOIS números:
  fora%   = retorno de "ficar fora" com esse custo (já deixa de ser 0)
  bh%     = comprar e segurar
O agente só mostrou algo se bater o bh% E o fora%, e se não ficar o tempo todo comprado (exp perto de 1 = só copiou o bh).
"""
from __future__ import annotations

import numpy as np

from udicoin.agente.experimentos_q import _bh, medir
from udicoin.agente.q_learning import AgenteQ, treinar
from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv
from udicoin.market.conta import ConfigConta
from udicoin.market.price_model import ConfigMercado


def _fixo(env, acao, seeds):
    r = []
    for s in seeds:
        obs, info = env.reset(seed=s)
        eq0, fim = info["patrimonio"], False
        while not fim:
            obs, _, _, fim, info = env.step(acao)
        r.append(100 * (info["patrimonio"] / eq0 - 1))
    return np.mean(r)


def main(periodos=None, mults=(1, 3), custos=(0.0, 0.0002, 0.0005), episodios=300, sementes=(0, 1)):
    def mk(cfg, custo, div):
        c = ConfigAmbiente(divisao=div, mercado=cfg, conta=ConfigConta(custo_caixa=custo))
        return UdiCoinEnv(periodos, c) if periodos else UdiCoinEnv.do_csv(c)

    base, seeds = ConfigMercado(), range(1000, 1030)
    print(f"{'tendência':>9} {'custo/dia':>10} | {'fora%':>7} {'bh%':>7} {'agente%':>8} {'exp':>5}")
    for m in mults:
        cfg = ConfigMercado(mu=tuple(x * m for x in base.mu))
        for custo in custos:
            et, ev = mk(cfg, custo, "treino"), mk(cfg, custo, "validacao")
            res = []
            for sm in sementes:
                ag = AgenteQ(alfa=0.02, gama=0.99, semente=sm)
                treinar(et, ag, episodios, semente=sm)
                res.append(medir(ev, ag, seeds))
            print(f"{m:8d}x {custo:10.4f} | {_fixo(ev, 0, seeds):+7.2f} {_fixo(ev, 2, seeds):+7.2f} "
                  f"{np.mean([x[0] for x in res]):+8.2f} {np.mean([x[1] for x in res]):5.2f}", flush=True)


if __name__ == "__main__":
    main()
