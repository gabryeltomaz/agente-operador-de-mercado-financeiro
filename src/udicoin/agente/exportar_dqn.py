"""Exporta a rede treinada para o site:  python -m udicoin.agente.exportar_dqn --nivel 5

Gera web/data/agente_dqn.js com os pesos (JSON dentro de `window.AGENTE_DQN = ...`) e alguns "vetores de teste"
(entrada -> notas calculadas aqui no PyTorch) para o JavaScript conferir que calcula a MESMA coisa.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from udicoin.agente.dqn import AgenteDQN
from udicoin.env.features import NOMES, calcular_features
from udicoin.env.trading_env import ConfigAmbiente

RAIZ = Path(__file__).resolve().parents[3]
SAIDA = RAIZ / "web" / "data" / "agente_dqn.js"


def _arredonda(x, casas=6):
    return np.round(np.asarray(x, dtype=float), casas).tolist()


# Hiperparâmetros do treino longo (iguais aos de treino_dqn_longo.py), mostrados na tela "Rede neural".
TREINO = {
    "algoritmo": "DQN (Deep Q-Network)",
    "gama": 0.99, "taxa_aprendizado": 0.001, "lote": 64, "memoria": 100_000, "rede_alvo_a_cada": 500,
    "passo_de_treino_a_cada": 2, "epsilon_inicial": 1.0, "epsilon_final": 0.05, "epsilon_cai_ate": 0.7,
    "recompensa": "log do retorno diário do patrimônio, já descontando taxa e slippage",
    "periodo_treino": "2010–2019 (dólar real como âncora)", "dias_por_episodio": 252,
}


def ler_curva(arquivo_pesos: Path | None) -> list[dict]:
    """Procura o CSV da curva (gravado pelo treino longo) ao lado dos pesos. Devolve [] se não achar."""
    if arquivo_pesos is None:
        return []
    nome = Path(arquivo_pesos).name
    for sufixo in ("_final.pt", "_melhor.pt"):
        if nome.endswith(sufixo):
            csv_ = Path(arquivo_pesos).with_name(nome[: -len(sufixo)] + "_curva.csv")
            if csv_.exists():
                with open(csv_, newline="", encoding="utf-8") as f:
                    return [{k: float(v) for k, v in linha.items()} for linha in csv.DictReader(f)]
    return []


def montar(agente: AgenteDQN, nivel: float, n_testes: int = 12, semente: int = 0, curva: list[dict] | None = None) -> dict:
    lineares = [m for m in agente.rede.camadas if isinstance(m, nn.Linear)]
    camadas = [{"W": _arredonda(m.weight.detach().numpy()), "b": _arredonda(m.bias.detach().numpy())} for m in lineares]
    rng = np.random.default_rng(semente)
    testes = []
    for _ in range(n_testes):
        closes = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.008, 21))), 4)
        expo, lucro = float(rng.choice([0.0, 0.5, 1.0])), float(rng.normal(0, 0.03))
        obs = calcular_features(closes, expo, lucro)
        testes.append({"closes": closes.tolist(), "exposicao": expo, "lucro": round(lucro, 6),
                       "features": _arredonda(obs), "notas": _arredonda(agente.notas(obs))})
    return {
        "nivel": nivel,  # força das tendências do mercado em que a rede foi treinada
        "niveis": list(ConfigAmbiente().niveis),  # exposição-alvo de cada ação
        "nomes_features": list(NOMES),
        "camadas": camadas,
        "testes": testes,
        "treino": {**TREINO, "episodios": int(max((c["episodio"] for c in curva), default=0)) or None},
        "curva": curva or [],
    }


def exportar(arquivo_pesos, nivel: float, saida: Path = SAIDA) -> Path:
    ag = AgenteDQN()
    ag.rede.load_state_dict(torch.load(arquivo_pesos))
    dados = montar(ag, nivel, curva=ler_curva(Path(arquivo_pesos)))
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text("// GERADO por udicoin.agente.exportar_dqn — não edite.\nwindow.AGENTE_DQN = "
                     + json.dumps(dados, separators=(",", ":")) + ";\n", encoding="utf-8")
    return saida


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nivel", type=float, default=5.0)
    p.add_argument("--pesos", default=None, help="arquivo .pt (padrão: pesos/dqn_<nivel>x_s0_final.pt)")
    a = p.parse_args()
    if a.pesos:
        arq = Path(a.pesos)
    else:
        cands = [RAIZ / "pesos" / f"dqn_{t}x_s0_final.pt" for t in (f"{a.nivel:g}", str(a.nivel))]
        arq = next((c for c in cands if c.exists()), None)
        if arq is None:
            raise SystemExit(f"Não achei os pesos: {cands[0]}")
    print("gerado:", exportar(arq, a.nivel))


if __name__ == "__main__":
    main()
