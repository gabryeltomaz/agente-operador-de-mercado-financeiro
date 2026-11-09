# Passo 0 — O ambiente como um MDP

> Documento para explicar ao professor (e a mim mesmo) **o que o agente vê, o que ele faz, o que ele ganha e por quê.**

## 1. Visão geral
Um agente negocia a **udiCOIN**, uma moeda fictícia cujo preço é ancorado no dólar real (USD/BRL).
Ele começa cada episódio com R$ 10.000 em caixa e, a cada dia útil, decide quanto do patrimônio quer manter em udiCOIN.
Objetivo: fazer o patrimônio crescer, pagando taxas e o impacto de suas próprias ordens.

Por que ancorar no dólar real? Para o mercado ter uma estrutura plausível (tendências e crises reais), sem ser um dado real
que o agente apenas decore. A moeda é inventada; o **valor justo V** (o dólar reescalado) é escondido do agente.

## 2. Como o preço é gerado (a "física")
```
ln(preço_t) = ln(V_t) + d_t
d_t = (1 − k)·d_{t−1} + μ[regime_t] + σ[regime_t]·ε_t + λ·fluxo_t / profundidade
```
- `V_t`: dólar real reescalado para começar em R$ 100. **Escondido.**
- `d_t`: desvio fictício; um elástico (`k = 0,02`) o puxa de volta para zero. Nos testes com dólar real, o preço fica
  tipicamente ~3% longe de V.
- **Regime escondido** (alta, baixa, lateral), uma cadeia de Markov que empurra `d`. **Escondido.**
- **Impacto:** o fluxo líquido que o agente negociou ontem (+ compra, − venda, em R$) empurra o preço de hoje.
- O modelo em Python foi conferido contra o site (JS) com o dólar real: todas as medidas bateram.

## 3. O MDP
| Elemento | Definição |
|---|---|
| **Estado (observação)** | 7 números públicos: `ret_1d`, `ret_5d`, `ret_20d` (retornos recentes), `razao_mm_5_20` (média de 5 dias / média de 20), `vol_20d` (volatilidade), `exposicao` (fração do patrimônio em udiCOIN), `lucro_posicao` (ganho não realizado). Escalas fixas, cortadas em ±3. |
| **O que o agente NÃO vê** | O valor justo V, o desvio `d`, o regime e o futuro. Por isso o ambiente é **parcialmente observável**: o agente tem de inferir o contexto pelas features. |
| **Ações** | 3 exposições-alvo: **0** (tudo em caixa), **0,5**, **1** (tudo em udiCOIN). |
| **Transição** | A conta ajusta a posição ao preço de fechamento de hoje (com taxa e slippage). O fluxo da ordem entra no preço de amanhã. O preço de amanhã vem do modelo da seção 2. |
| **Recompensa** | `r_t = ln(patrimônio_{t+1} / patrimônio_t)`, já líquida de taxa e slippage. |
| **Episódio** | 252 dias úteis (1 ano), depois de 20 dias de aquecimento só para formar o histórico. Termina por limite de tempo. |
| **Dados** | Treino 2010–2019, validação 2020–2022, teste 2023 em diante. A janela de cada episódio é sorteada dentro da divisão. |

## 4. Custos do agente
- Taxa: 0,1% do valor negociado.
- Slippage: meio spread de 0,02% + `λ · valor / profundidade` (ordem maior paga mais).
- Banda sem operar: se a exposição já está a menos de 2 pontos percentuais do alvo, não opera (alvo 0 sempre zera).
- Sem alavancagem e sem venda a descoberto. Identidade verificada por teste: o patrimônio só cai o valor dos custos.

## 5. Por que essa recompensa?
1. **Log-retorno soma:** `Σ r_t = ln(patrimônio final / inicial)`. Maximizar a soma é maximizar o patrimônio final, sem inventar um
   objetivo diferente do que queremos.
2. **Penaliza perdas profundas:** perder 50% exige ganhar 100% para voltar. O log trata isso de forma assimétrica e natural, o que
   desencoraja apostas insanas sem eu precisar escrever uma penalidade de risco à mão.
3. **Custos entram automaticamente:** operar demais reduz a recompensa, então o agente aprende a não girar a carteira à toa.
4. **Recompensa densa:** há um sinal todo dia (não só no fim do ano), o que ajuda o aprendizado.
5. Ela se adapta depois a **perfis** (agressivo, moderado, seguro) acrescentando pesos de risco, sem mudar o resto do ambiente.

## 6. Por que essas ações?
- **Exposição-alvo em vez de "comprar/vender X":** toda ação é sempre válida (não existe vender o que não se tem), então o agente
  nunca desperdiça aprendizado com ações ilegais.
- **Poucos níveis (3):** mantém a tabela Q viável no começo e a rede pequena depois; dá para aumentar.
- A ação é um *alvo*, não uma ordem: segurar a mesma ação não gera custo.

## 7. Decisões de honestidade (o que vou dizer ao professor)
- É uma **simulação**: o que funciona aqui pode não funcionar no mercado real.
- Com dólar subindo no treino, "comprar e segurar" ganha em média. O agente precisa ser comparado a isso, e a avaliação usa também
  validação e teste (períodos diferentes).
- Um agente que **veja** o desvio (teste de sanidade, o "trapaceiro") deve ganhar do que não vê; se não ganhar, há bug no ambiente.
- O agente pode terminar sem bater o comprado e parado. Isso é um resultado válido, desde que bem explicado.

## 8. Referências medidas (20 episódios, treino, dólar real)
| Política | Retorno médio | Pior | Melhor |
|---|---|---|---|
| Sempre zerada | 0,00% | 0,0% | 0,0% |
| Sempre comprada | 12,25% | −21,6% | 44,4% |
| Aleatória | −10,26% | −24,3% | 8,3% |

A aleatória perde ~10% ao ano só de custos por operar toda hora.

## 9. Como reproduzir
```
pytest -q
python -m udicoin.env.demo
python -m udicoin.env.figuras     # gera docs/figuras/*.png
```
