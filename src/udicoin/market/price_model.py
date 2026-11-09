"""Passo 0.2: o modelo de preço da udiCOIN (a "física" do mercado).

Em uma frase: a udiCOIN é o dólar real (reescalado para começar em R$ 100) vezes um desvio
fictício que um elástico suave puxa de volta para zero.

    ln(preço_t) = ln(V_t) + d_t

    V_t  = valor justo = dólar real reescalado.   ESCONDIDO: o agente nunca vê.
    d_t  = (1 - k)·d_{t-1}  +  μ[regime_t]  +  σ[regime_t]·ε_t  +  λ·fluxo / profundidade
             │                  │              │                    └ impacto das ordens
             │                  │              └ ruído (ε ~ Normal(0, 1))
             │                  └ deriva do regime escondido (alta, baixa ou lateral)
             └ elástico: devolve a fração k do desvio a cada dia

O regime muda por uma cadeia de Markov (fica muitos dias no mesmo estado).
O `fluxo` é o líquido de compras menos vendas (em R$) que chega NESTE passo, isto é, o
que foi negociado no passo anterior. Assim a ordem de hoje só mexe no preço de amanhã.

É o mesmo modelo do site (data-provider.js), sem o arredondamento de centavos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

REGIMES = ("alta", "baixa", "lateral")
ALTA, BAIXA, LATERAL = 0, 1, 2


@dataclass(frozen=True)
class ConfigMercado:
    p0: float = 100.0  # preço inicial da udiCOIN (R$)
    k: float = 0.02  # força do elástico (meia-vida ≈ ln2/k ≈ 34 dias úteis)
    mu: tuple[float, float, float] = (0.0012, -0.0012, 0.0)  # deriva do desvio por regime
    sigma: tuple[float, float, float] = (0.004, 0.0052, 0.0032)  # ruído diário por regime
    # linha = regime de hoje; coluna = regime de amanhã (alta, baixa, lateral)
    transicao: tuple[tuple[float, float, float], ...] = (
        (0.965, 0.010, 0.025),
        (0.010, 0.955, 0.035),
        (0.035, 0.025, 0.940),
    )
    sigma_ref: float = 0.008  # volatilidade diária de referência (volume sintético)
    pavio: float = 0.004  # tamanho dos pavios das velas
    volume_base: float = 1000.0
    impacto_lambda: float = 1.0
    impacto_profundidade: float = 2e7  # "tamanho do mercado" em R$/dia: maior = menos impacto

    def __post_init__(self) -> None:
        if not self.p0 > 0:
            raise ValueError("p0 deve ser positivo")
        if not 0.0 <= self.k <= 1.0:
            raise ValueError("k deve estar entre 0 e 1")
        if len(self.mu) != 3 or len(self.sigma) != 3:
            raise ValueError("mu e sigma precisam ter 3 valores (alta, baixa, lateral)")
        if any(s < 0 for s in self.sigma):
            raise ValueError("sigma não pode ser negativo")
        m = np.array(self.transicao, dtype=float)
        if m.shape != (3, 3) or (m < 0).any() or not np.allclose(m.sum(axis=1), 1.0):
            raise ValueError("transicao deve ser 3x3, sem negativos, com cada linha somando 1")
        if not self.impacto_profundidade > 0:
            raise ValueError("impacto_profundidade deve ser positivo")


def distribuicao_estacionaria(transicao: np.ndarray) -> np.ndarray:
    """Fração de tempo que a cadeia passa em cada regime no longo prazo."""
    m = np.asarray(transicao, dtype=float)
    n = len(m)
    a = np.vstack([m.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    return np.linalg.lstsq(a, b, rcond=None)[0]


@dataclass(frozen=True)
class Tick:
    """O que é PÚBLICO: o agente pode (e só pode) usar isto."""

    t: int
    abertura: float
    maxima: float
    minima: float
    fechamento: float
    volume: float


@dataclass(frozen=True)
class Oculto:
    """O que é ESCONDIDO: só para análise e testes. Nunca vai para a observação do agente."""

    ancora: float  # valor justo V_t (em R$ da udiCOIN)
    desvio: float  # d_t
    regime: int  # 0 alta, 1 baixa, 2 lateral


class ProcessoDePreco:
    """Gera o preço dia a dia. É o coração do ambiente.

    ln_ancora: logaritmo da série real do dólar para o episódio (N+1 valores, N passos).
    O processo usa só as DIFERENÇAS entre valores consecutivos: o dólar entra como
    retornos, e o preço começa sempre em `p0`.
    """

    def __init__(
        self,
        ln_ancora: np.ndarray,
        cfg: ConfigMercado | None = None,
        rng: np.random.Generator | None = None,
        regime_inicial: int | str | None = None,
    ) -> None:
        ln = np.asarray(ln_ancora, dtype=float)
        if ln.ndim != 1 or len(ln) < 2:
            raise ValueError("ln_ancora deve ser um vetor com pelo menos 2 valores")
        if not np.isfinite(ln).all():
            raise ValueError("ln_ancora tem valores não finitos")
        self.ln_ancora = ln
        self.cfg = cfg or ConfigMercado()
        self.rng = rng if rng is not None else np.random.default_rng()
        self._trans = np.array(self.cfg.transicao, dtype=float)
        self._acum = np.cumsum(self._trans, axis=1)
        self._estacionaria = distribuicao_estacionaria(self._trans)
        self.reset(regime_inicial)

    @property
    def tamanho(self) -> int:
        """Quantos instantes (ticks) a âncora permite: t = 0 .. tamanho-1."""
        return len(self.ln_ancora)

    def reset(self, regime_inicial: int | str | None = None) -> None:
        c = self.cfg
        self.t = -1  # ainda nenhum tick gerado
        self.ln_v = math.log(c.p0)
        self.d = 0.0
        self.fechamento_ant = c.p0
        if regime_inicial is None:
            self.regime = int(self.rng.choice(3, p=self._estacionaria))
        elif isinstance(regime_inicial, str):
            self.regime = REGIMES.index(regime_inicial)
        else:
            self.regime = int(regime_inicial)
        if self.regime not in (0, 1, 2):
            raise ValueError("regime inicial deve ser 0, 1, 2 ou o nome do regime")

    def _proximo_regime(self) -> int:
        u = self.rng.random()
        return min(int(np.searchsorted(self._acum[self.regime], u, side="right")), 2)

    def step(self, fluxo_brl: float = 0.0) -> tuple[Tick, Oculto]:
        """Avança um dia. `fluxo_brl` = compras - vendas (R$) negociadas no dia anterior."""
        t = self.t + 1
        ln = self.ln_ancora
        if t >= len(ln):
            raise IndexError("acabou a série da âncora")
        c = self.cfg
        if t > 0:
            self.regime = self._proximo_regime()
            self.ln_v += float(ln[t] - ln[t - 1])  # o dólar real anda como andou
        impacto = c.impacto_lambda * fluxo_brl / c.impacto_profundidade
        eps = float(self.rng.standard_normal())
        self.d = (1.0 - c.k) * self.d + c.mu[self.regime] + c.sigma[self.regime] * eps + impacto
        fechamento = math.exp(self.ln_v + self.d)
        abertura = self.fechamento_ant
        retorno = math.log(fechamento / abertura)
        g = np.abs(self.rng.standard_normal(2))
        maxima = max(abertura, fechamento) * (1.0 + float(g[0]) * c.pavio)
        minima = min(abertura, fechamento) * (1.0 - float(g[1]) * c.pavio)
        volume = c.volume_base * (1.0 + abs(retorno) / c.sigma_ref) * (0.7 + 0.6 * float(self.rng.random()))
        self.t = t
        self.fechamento_ant = fechamento
        return (
            Tick(t, abertura, maxima, minima, fechamento, volume),
            Oculto(math.exp(self.ln_v), self.d, self.regime),
        )


def simular(
    ln_ancora: np.ndarray,
    cfg: ConfigMercado | None = None,
    semente: int = 0,
    fluxos: np.ndarray | None = None,
    regime_inicial: int | str | None = None,
) -> dict[str, np.ndarray]:
    """Roda o processo inteiro de uma vez. Útil para gráficos, estatísticas e testes.

    `fluxos[t]` é o fluxo líquido (R$) aplicado no instante t (por padrão, zero).
    """
    ln = np.asarray(ln_ancora, dtype=float)
    n = len(ln)
    fl = np.zeros(n) if fluxos is None else np.asarray(fluxos, dtype=float)
    if len(fl) != n:
        raise ValueError("fluxos deve ter o mesmo tamanho de ln_ancora")
    proc = ProcessoDePreco(ln, cfg, np.random.default_rng(semente), regime_inicial)
    saida = {nome: np.empty(n) for nome in ("abertura", "maxima", "minima", "close", "volume", "ancora", "desvio")}
    saida["regime"] = np.empty(n, dtype=int)
    for i in range(n):
        tick, oculto = proc.step(fl[i])
        saida["abertura"][i] = tick.abertura
        saida["maxima"][i] = tick.maxima
        saida["minima"][i] = tick.minima
        saida["close"][i] = tick.fechamento
        saida["volume"][i] = tick.volume
        saida["ancora"][i] = oculto.ancora
        saida["desvio"][i] = oculto.desvio
        saida["regime"][i] = oculto.regime
    return saida
