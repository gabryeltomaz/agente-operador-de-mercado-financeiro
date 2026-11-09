"""Confere se o modelo de preço em Python se comporta como o do site (JavaScript).

Os dois usam geradores aleatórios diferentes, então os números exatos não coincidem. O que
comparamos é a ESTATÍSTICA: média sobre várias sementes do lado Python contra os valores que
saíram do console do site (2120 passos a partir da primeira linha do CSV, com a mesma série).

Uso (com o CSV real em data/usdbrl.csv):
    python -m udicoin.market.paridade
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from udicoin.data.loader import carregar_dolar
from udicoin.market.diagnostico import metricas_ancora
from udicoin.market.price_model import ConfigMercado, simular

# Valores medidos no console do site (anchorStart='first', 120 de aquecimento + 2000 passos).
REFERENCIA_JS: dict[float, dict[str, float]] = {
    0.005: {"dp_pct": 6.32, "max_pct": 20.1, "corr": 0.89, "vol_udi_pct": 0.97, "vol_dolar_pct": 0.86},
    0.02: {"dp_pct": 3.14, "max_pct": 11.1, "corr": 0.89, "vol_udi_pct": 0.97, "vol_dolar_pct": 0.86},
    0.05: {"dp_pct": 1.83, "max_pct": 6.5, "corr": 0.89, "vol_udi_pct": 0.97, "vol_dolar_pct": 0.86},
    0.1: {"dp_pct": 1.2, "max_pct": 4.3, "corr": 0.89, "vol_udi_pct": 0.97, "vol_dolar_pct": 0.86},
}
# Tolerância mínima por medida. A tolerância final é o maior entre esta e 3 desvios-padrão
# entre sementes (o valor do site é UMA realização, então varia).
TOLERANCIA_MIN = {"dp_pct": 0.5, "max_pct": 3.0, "corr": 0.03, "vol_udi_pct": 0.06, "vol_dolar_pct": 0.01}
PASSOS = 2120


def comparar(df: pd.DataFrame, n_sementes: int = 40, passos: int = PASSOS) -> list[dict]:
    """Devolve uma linha por (k, medida) com média/desvio do Python, referência do site e veredito."""
    if len(df) < passos:
        raise ValueError(f"a série tem {len(df)} linhas, mas a comparação usa {passos}")
    ln = np.log(df["fechamento"].to_numpy(dtype=float))[:passos]
    linhas = []
    for k, ref in REFERENCIA_JS.items():
        amostras: dict[str, list[float]] = {nome: [] for nome in ref}
        for semente in range(n_sementes):
            r = simular(ln, ConfigMercado(k=k), semente=semente)
            for nome, valor in metricas_ancora(r["close"], r["ancora"]).items():
                amostras[nome].append(valor)
        for nome, valores in amostras.items():
            media, dp = float(np.mean(valores)), float(np.std(valores))
            tol = max(TOLERANCIA_MIN[nome], 3 * dp)
            linhas.append(
                {
                    "k": k, "medida": nome, "python_media": media, "python_dp": dp,
                    "js": ref[nome], "tolerancia": tol, "ok": abs(media - ref[nome]) <= tol,
                }
            )
    return linhas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="caminho do CSV (padrão: data/usdbrl.csv)")
    ap.add_argument("--sementes", type=int, default=40)
    a = ap.parse_args()
    df = carregar_dolar(a.csv)
    print(f"Série: {len(df)} dias, de {df['data'].iloc[0].date()} a {df['data'].iloc[-1].date()}")
    linhas = comparar(df, a.sementes)
    print(f"\n{'k':>6}  {'medida':<14} {'Python (média ± dp)':>22} {'site (JS)':>10} {'tolerância':>11}  ok")
    for l in linhas:
        print(
            f"{l['k']:>6}  {l['medida']:<14} {l['python_media']:>14.2f} ± {l['python_dp']:<5.2f} "
            f"{l['js']:>10.2f} {l['tolerancia']:>11.2f}  {'✓' if l['ok'] else '✗'}"
        )
    falhas = [l for l in linhas if not l["ok"]]
    print("\nRESULTADO:", "tudo bate com o site" if not falhas else f"{len(falhas)} medida(s) fora da tolerância")
    return 0 if not falhas else 1


if __name__ == "__main__":
    sys.exit(main())
