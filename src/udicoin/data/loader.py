"""Passo 0.1: leitura, validação e divisão do histórico do dólar.

O ambiente usa esta série como o "valor justo" ESCONDIDO da udiCOIN. Aqui só garantimos
que os dados estão limpos e separamos treino, validação e teste por data, para o agente
nunca ser avaliado em dias que viu no treino.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
CAMINHO_PADRAO = RAIZ / "data" / "usdbrl.csv"

# (início, fim), ambos inclusivos. fim=None significa "até o último dia disponível".
DIVISOES: dict[str, tuple[str, str | None]] = {
    "treino": ("2010-01-01", "2019-12-31"),
    "validacao": ("2020-01-01", "2022-12-31"),
    "teste": ("2023-01-01", None),
}


class DadosInvalidos(ValueError):
    """O arquivo de dados não serve para a simulação."""


def carregar_dolar(
    caminho: str | Path | None = None, min_linhas: int = 500, salto_aviso: float = 0.15
) -> pd.DataFrame:
    """Lê o CSV (colunas `data`, `fechamento`) e valida. Devolve um DataFrame limpo.

    Avisos não fatais (saltos grandes entre dias) ficam em `df.attrs["avisos"]`.
    """
    caminho = Path(caminho) if caminho is not None else CAMINHO_PADRAO
    if not caminho.exists():
        raise FileNotFoundError(
            f"Não achei {caminho}. Gere o arquivo com: py tools\\baixar_dolar.py"
        )
    df = pd.read_csv(caminho)
    faltando = {"data", "fechamento"} - set(df.columns)
    if faltando:
        raise DadosInvalidos(f"faltam as colunas {sorted(faltando)} (achei {list(df.columns)})")
    df = df[["data", "fechamento"]].copy()
    try:
        df["data"] = pd.to_datetime(df["data"], format="%Y-%m-%d")
    except (ValueError, TypeError) as erro:
        raise DadosInvalidos(f"datas fora do formato AAAA-MM-DD: {erro}") from erro
    df["fechamento"] = pd.to_numeric(df["fechamento"], errors="coerce")
    if df["fechamento"].isna().any():
        raise DadosInvalidos("há valores de fechamento vazios ou não numéricos")
    if (df["fechamento"] <= 0).any():
        raise DadosInvalidos("há valores de fechamento menores ou iguais a zero")
    if df["data"].duplicated().any():
        raise DadosInvalidos("há datas repetidas")
    if not df["data"].is_monotonic_increasing:
        raise DadosInvalidos("as datas não estão em ordem crescente")
    if len(df) < min_linhas:
        raise DadosInvalidos(f"poucos dados: {len(df)} linhas (mínimo {min_linhas})")

    saltos = df["fechamento"].pct_change().abs()
    avisos = [
        f"salto de {saltos[i]:.0%} em {df['data'][i].date()}" for i in saltos.index[saltos > salto_aviso]
    ]
    df = df.reset_index(drop=True)
    df.attrs["avisos"] = avisos
    return df


@dataclass(frozen=True, eq=False)
class Periodo:
    """Um pedaço do histórico (treino, validação ou teste)."""

    nome: str
    datas: np.ndarray  # datetime64
    fechamento: np.ndarray  # float64

    def __len__(self) -> int:
        return len(self.fechamento)

    @property
    def ln(self) -> np.ndarray:
        """Logaritmo natural do fechamento. O modelo de preço só usa as DIFERENÇAS dele."""
        return np.log(self.fechamento)


def dividir(df: pd.DataFrame, nome: str) -> Periodo:
    if nome not in DIVISOES:
        raise KeyError(f"divisão desconhecida: {nome!r}. Use uma de {list(DIVISOES)}")
    inicio, fim = DIVISOES[nome]
    mascara = df["data"] >= pd.Timestamp(inicio)
    if fim is not None:
        mascara &= df["data"] <= pd.Timestamp(fim)
    sub = df[mascara]
    if sub.empty:
        raise DadosInvalidos(f"a divisão {nome!r} ({inicio} a {fim or 'hoje'}) não tem dados")
    return Periodo(nome, sub["data"].to_numpy(), sub["fechamento"].to_numpy(dtype=float))


def periodos(df: pd.DataFrame) -> dict[str, Periodo]:
    return {nome: dividir(df, nome) for nome in DIVISOES}


def sortear_inicio(periodo: Periodo, passos: int, rng: np.random.Generator) -> int:
    """Sorteia onde começa um episódio de `passos` passos, sem sair do período.

    Um episódio de N passos usa N+1 linhas: a âncora do instante 0 mais um retorno por passo.
    """
    linhas = passos + 1
    if len(periodo) < linhas:
        raise DadosInvalidos(
            f"o período {periodo.nome!r} tem {len(periodo)} linhas, mas um episódio de {passos} passos precisa de {linhas}"
        )
    return int(rng.integers(0, len(periodo) - linhas + 1))


def janela_ln(periodo: Periodo, inicio: int, passos: int) -> np.ndarray:
    """O log da âncora para um episódio: N+1 valores a partir de `inicio`."""
    return periodo.ln[inicio : inicio + passos + 1]
