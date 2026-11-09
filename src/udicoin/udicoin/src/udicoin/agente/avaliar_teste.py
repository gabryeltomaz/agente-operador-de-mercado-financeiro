"""AVALIAÇÃO FINAL no período de TESTE (2023 em diante). Rode UMA ÚNICA VEZ.

    python -m udicoin.agente.avaliar_teste --nivel 5

Regras fixadas ANTES de ver o resultado (por isso ele é honesto):
  - usa o DQN `final` (não o "melhor", que sofre de maldição do vencedor);
  - 200 episódios com sementes 9000-9199 (nunca usadas em treino nem validação);
  - compara com tabela Q (treinada igual), comprar e segurar e ficar fora;
  - relata o cenário em que o agente foi treinado (--nivel, padrão 5x) E o cenário realista (1x),
    com os mesmos pesos, sem retreinar;
  - nada é ajustado depois de ver o número.
O resultado vai para resultados/teste_final.json e a trava resultados/teste_final.TRAVA impede repetir.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from udicoin.agente.avaliar_dqn import episodios
from udicoin.agente.dqn import AgenteDQN
from udicoin.agente.q_learning import AgenteQ, discretizar, treinar
from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv
from udicoin.market.price_model import ConfigMercado

RAIZ = Path(__file__).resolve().parents[3]


def cfg_nivel(nivel):
    return ConfigMercado(mu=tuple(x * nivel for x in ConfigMercado().mu))


def achar_pesos(pasta: Path, nivel: float, sem: int = 0) -> Path:
    for t in (f"{nivel:g}", str(nivel)):
        arq = pasta / f"dqn_{t}x_s{sem}_final.pt"
        if arq.exists():
            return arq
    raise FileNotFoundError(f"não achei os pesos finais do nível {nivel:g}x em {pasta}")


def main(nivel=5.0, n=200, forcar=False):
    saida = RAIZ / "resultados"; saida.mkdir(exist_ok=True)
    trava = saida / "teste_final.TRAVA"
    if trava.exists() and not forcar:
        print("A avaliação de teste JÁ foi feita (veja resultados/teste_final.json). Repetir invalida o teste.")
        return
    ag = AgenteDQN()
    ag.rede.load_state_dict(torch.load(achar_pesos(RAIZ / "pesos", nivel)))
    et = UdiCoinEnv.do_csv(ConfigAmbiente(divisao="treino", mercado=cfg_nivel(nivel)))
    tab = AgenteQ(alfa=0.02, gama=0.99, semente=0)
    treinar(et, tab, 300, semente=0)
    seeds = range(9000, 9000 + n)
    res = {"data": str(date.today()), "episodios": n, "pesos": "DQN final", "cenarios": {}}
    for nome, nv in ((f"{nivel:g}x (cenário de treino)", nivel), ("1x (cenário realista, sem retreinar)", 1.0)):
        ev = UdiCoinEnv.do_csv(ConfigAmbiente(divisao="teste", mercado=cfg_nivel(nv)))
        m = {
            "comprar e segurar": episodios(ev, lambda o: 2, seeds),
            "ficar fora": episodios(ev, lambda o: 0, seeds),
            "tabela Q": episodios(ev, lambda o: tab.agir(discretizar(o), 0.0), seeds),
            "DQN": episodios(ev, lambda o: ag.agir(o, 0.0), seeds),
        }
        bh = m["comprar e segurar"][0]
        print(f"\n=== TESTE (2023+) · tendência {nome} · {n} episódios ===")
        print(f"{'método':>18} | {'retorno% ± erro padrão':>22} | {'exp':>5} | {'ganha do b&h':>12}")
        res["cenarios"][nome] = {}
        for k, (r, e) in m.items():
            ep = r.std() / np.sqrt(len(r))
            print(f"{k:>18} | {r.mean():+9.2f} ± {ep:4.2f}       | {e.mean():5.2f} | {np.mean(r > bh) * 100:10.0f}%")
            res["cenarios"][nome][k] = {"media": float(r.mean()), "erro_padrao": float(ep), "exposicao": float(e.mean()),
                                        "ganha_bh_pct": float(np.mean(r > bh) * 100)}
    (saida / "teste_final.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    trava.write_text("teste final já usado em " + res["data"], encoding="utf-8")
    print("\nSalvo em resultados/teste_final.json. Não rode de novo e não ajuste nada com base nisto.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nivel", type=float, default=5.0)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--forcar", action="store_true", help="ignora a trava (NÃO use: invalida o teste)")
    a = p.parse_args()
    main(a.nivel, a.n, a.forcar)
