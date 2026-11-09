"""Passo 0.3: a conta do agente (caixa, posição, taxa e slippage).

Regras (simples de propósito, sem alavancagem e sem venda a descoberto):

* O agente pede uma EXPOSIÇÃO-ALVO entre 0 e 1 (fração do patrimônio aplicada em udiCOIN).
* A conta compra ou vende o necessário para chegar lá, ao preço de fechamento do dia.
* Cada ordem paga:  taxa  (fração do valor negociado)  +  slippage
      slippage = meio_spread + λ · valor / profundidade      (o mesmo λ e a mesma profundidade do preço)
  Comprar paga um preço um pouco MAIOR que o fechamento; vender, um pouco MENOR.
* Banda sem operar: se a exposição atual já está a menos de `banda` do alvo, não faz nada
  (evita pagar custo por ajustes minúsculos). Alvo 0 sempre zera a posição.

Identidade contábil (testada): patrimônio_depois = patrimônio_antes − (taxa + custo de slippage).
O fluxo (+ compra, − venda, em R$) é o que o ambiente entrega ao modelo de preço para o dia seguinte.
"""
from __future__ import annotations

from dataclasses import dataclass

from udicoin.market.price_model import ConfigMercado

_SLIP_MAX = 0.5


@dataclass(frozen=True)
class ConfigConta:
    capital_inicial: float = 10_000.0
    taxa: float = 0.001  # 0,1% do valor negociado
    meio_spread: float = 0.0002  # 0,02% para cada lado
    banda: float = 0.02  # não opera se |alvo − exposição| < 2 pontos percentuais
    custo_caixa: float = 0.0  # perda diária do dinheiro parado (ex.: 0.0002 ≈ 5% ao ano). 0 = desligado

    def __post_init__(self) -> None:
        if not self.capital_inicial > 0:
            raise ValueError("capital_inicial deve ser positivo")
        if not 0 <= self.taxa < 1 or not 0 <= self.meio_spread < 1:
            raise ValueError("taxa e meio_spread devem estar em [0, 1)")
        if self.banda < 0:
            raise ValueError("banda não pode ser negativa")
        if not 0 <= self.custo_caixa < 1:
            raise ValueError("custo_caixa deve estar em [0, 1)")


@dataclass(frozen=True)
class Execucao:
    tipo: str  # "compra", "venda" ou "nada"
    quantidade: float = 0.0  # unidades de udiCOIN
    valor_ref: float = 0.0  # valor da ordem ao preço de fechamento (R$)
    preco_exec: float = 0.0
    taxa: float = 0.0  # R$
    custo_slippage: float = 0.0  # R$

    @property
    def custo(self) -> float:
        return self.taxa + self.custo_slippage

    @property
    def fluxo(self) -> float:
        """Fluxo líquido que empurra o preço: + compra, − venda (R$)."""
        return {"compra": self.valor_ref, "venda": -self.valor_ref}.get(self.tipo, 0.0)


NADA = Execucao("nada")


class Conta:
    def __init__(self, cfg_conta: ConfigConta | None = None, cfg_mercado: ConfigMercado | None = None) -> None:
        self.cfg = cfg_conta or ConfigConta()
        self.mercado = cfg_mercado or ConfigMercado()
        self.caixa = self.cfg.capital_inicial
        self.unidades = 0.0
        self.custo_medio = 0.0

    def passar_dia(self) -> None:
        """Fim do dia: o caixa parado perde `custo_caixa` (custo de oportunidade). A udiCOIN não sofre isso."""
        self.caixa *= 1.0 - self.cfg.custo_caixa

    def patrimonio(self, preco: float) -> float:
        return self.caixa + self.unidades * preco

    def exposicao(self, preco: float) -> float:
        return self.unidades * preco / self.patrimonio(preco)

    def lucro_posicao(self, preco: float) -> float:
        """Ganho não realizado da posição aberta (fração do custo médio); 0 se não há posição."""
        return preco / self.custo_medio - 1.0 if self.unidades > 0 else 0.0

    def _slip(self, valor: float) -> float:
        m = self.mercado
        return min(self.cfg.meio_spread + m.impacto_lambda * valor / m.impacto_profundidade, _SLIP_MAX)

    def ajustar_para(self, alvo: float, preco: float) -> Execucao:
        if not 0.0 <= alvo <= 1.0:
            raise ValueError("alvo de exposição deve estar em [0, 1]")
        if not preco > 0:
            raise ValueError("preço deve ser positivo")
        eq = self.patrimonio(preco)
        atual = self.unidades * preco
        dif = alvo * eq - atual
        zerar = alvo == 0.0 and self.unidades > 0
        if not zerar and abs(alvo - atual / eq) < self.cfg.banda:
            return NADA
        if zerar:
            dif = -atual
        if dif > 0:  # compra
            gasto = min(self.caixa, dif)
            if gasto <= 0:
                return NADA
            px = preco * (1.0 + self._slip(gasto))
            taxa = gasto * self.cfg.taxa
            qtd = (gasto - taxa) / px
            self.custo_medio = (self.unidades * self.custo_medio + qtd * px) / (self.unidades + qtd)
            self.unidades += qtd
            self.caixa -= gasto
            if self.caixa < 1e-9:
                self.caixa = 0.0
            return Execucao("compra", qtd, gasto, px, taxa, qtd * (px - preco))
        if dif < 0:  # venda
            valor = min(-dif, atual)
            qtd = valor / preco
            px = preco * (1.0 - self._slip(valor))
            bruto = qtd * px
            taxa = bruto * self.cfg.taxa
            self.caixa += bruto - taxa
            self.unidades = 0.0 if zerar or self.unidades - qtd < 1e-12 else self.unidades - qtd
            if self.unidades == 0.0:
                self.custo_medio = 0.0
            return Execucao("venda", qtd, valor, px, taxa, qtd * (preco - px))
        return NADA
