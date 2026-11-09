"""Passo 0.5: o ambiente (MDP) do agente de trading na udiCOIN.

Estado:    features públicas (preço passado + a própria conta). Nunca o valor justo nem o regime.
Ação:      índice em `niveis` = exposição-alvo (0, 0,5 ou 1). Qualquer ação é válida.
Transição: a conta executa no fechamento do dia t (com taxa e slippage); o fluxo da ordem entra
           no preço do dia t+1 (impacto); o preço de t+1 vem do modelo de preço.
Recompensa: ln(patrimônio_{t+1} / patrimônio_t), líquida de custos. A soma ao longo do episódio
           é exatamente ln(patrimônio final / inicial).
Episódio:  `passos` dias úteis (252) depois de `aquecimento` dias só para formar o histórico.
           Termina por truncamento (limite de tempo), nunca por "game over".

Interface parecida com a do Gymnasium: reset() -> (obs, info); step(a) -> (obs, r, terminated, truncated, info).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from udicoin.data.loader import Periodo, carregar_dolar, janela_ln, periodos as _periodos, sortear_inicio
from udicoin.env.features import LOOKBACK, NOME_TRAPACA, NOMES, calcular_features
from udicoin.market.conta import Conta, ConfigConta
from udicoin.market.price_model import ConfigMercado, ProcessoDePreco


@dataclass(frozen=True)
class ConfigAmbiente:
    passos: int = 252
    aquecimento: int = LOOKBACK
    niveis: tuple[float, ...] = (0.0, 0.5, 1.0)
    divisao: str = "treino"  # treino | validacao | teste
    trapaceiro: bool = False  # True: a observação inclui o desvio (SÓ para teste de sanidade)
    mercado: ConfigMercado = field(default_factory=ConfigMercado)
    conta: ConfigConta = field(default_factory=ConfigConta)

    def __post_init__(self) -> None:
        if self.passos < 1:
            raise ValueError("passos deve ser >= 1")
        if self.aquecimento < LOOKBACK:
            raise ValueError(f"aquecimento deve ser >= {LOOKBACK}")
        if not self.niveis or any(not 0.0 <= n <= 1.0 for n in self.niveis):
            raise ValueError("niveis devem estar em [0, 1]")


class UdiCoinEnv:
    def __init__(self, periodos: dict[str, Periodo], cfg: ConfigAmbiente | None = None) -> None:
        self.cfg = cfg or ConfigAmbiente()
        if self.cfg.divisao not in periodos:
            raise KeyError(f"divisão desconhecida: {self.cfg.divisao!r}")
        self.periodo = periodos[self.cfg.divisao]
        self.n_acoes = len(self.cfg.niveis)
        self.nomes_obs = NOMES + ((NOME_TRAPACA,) if self.cfg.trapaceiro else ())
        self.dim_obs = len(self.nomes_obs)
        self._rng = np.random.default_rng()
        self._pronto = False

    @classmethod
    def do_csv(cls, cfg: ConfigAmbiente | None = None, caminho=None) -> "UdiCoinEnv":
        return cls(_periodos(carregar_dolar(caminho)), cfg)

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        preco = self._fech[-1]
        return calcular_features(
            np.array(self._fech),
            self.conta.exposicao(preco),
            self.conta.lucro_posicao(preco),
            self._oculto.desvio if self.cfg.trapaceiro else None,
        )

    @property
    def historico_fechamentos(self) -> np.ndarray:
        return np.array(self._fech)

    def reset(self, seed: int | None = None, inicio: int | None = None, regime_inicial=None):
        c = self.cfg
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        n_janela = c.passos + c.aquecimento
        if inicio is None:
            inicio = sortear_inicio(self.periodo, n_janela, self._rng)
        self.inicio = inicio
        ln = janela_ln(self.periodo, inicio, n_janela)
        self.proc = ProcessoDePreco(ln, c.mercado, self._rng, regime_inicial)
        self.conta = Conta(c.conta, c.mercado)
        self._fech: list[float] = []
        for _ in range(c.aquecimento + 1):  # aquecimento: ninguém negocia
            tick, self._oculto = self.proc.step(0.0)
            self._fech.append(tick.fechamento)
        self.passo = 0
        self._pronto = True
        return self._obs(), self._info(None, None)

    def _info(self, acao, exe) -> dict:
        preco = self._fech[-1]
        t = self.proc.t
        return {
            "passo": self.passo,
            "fechamento": preco,
            "patrimonio": self.conta.patrimonio(preco),
            "exposicao": self.conta.exposicao(preco),
            "lucro_posicao": self.conta.lucro_posicao(preco),
            "acao": acao,
            "execucao": exe,
            "oculto": self._oculto,  # só para análise: nunca entra na observação
            "data_real": self.periodo.datas[self.inicio + t],
        }

    def step(self, acao: int):
        if not self._pronto:
            raise RuntimeError("chame reset() antes de step()")
        if isinstance(acao, bool) or not isinstance(acao, (int, np.integer)) or not 0 <= acao < self.n_acoes:
            raise ValueError(f"ação inválida: {acao!r} (use 0..{self.n_acoes - 1})")
        preco = self._fech[-1]
        eq0 = self.conta.patrimonio(preco)
        exe = self.conta.ajustar_para(self.cfg.niveis[int(acao)], preco)
        tick, self._oculto = self.proc.step(exe.fluxo)  # a ordem de hoje move o preço de amanhã
        self._fech.append(tick.fechamento)
        self.conta.passar_dia()  # o caixa parado perde valor (se custo_caixa > 0)
        eq1 = self.conta.patrimonio(tick.fechamento)
        self.passo += 1
        truncado = self.passo >= self.cfg.passos
        if truncado:
            self._pronto = False
        recompensa = math.log(eq1 / eq0)
        return self._obs(), recompensa, False, truncado, self._info(int(acao), exe)
