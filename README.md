# udiCOIN Market

Simulador de bolsa com uma moeda fictícia (**udiCOIN**) e bots de IA.
Stack do site: HTML + Tailwind (CDN) + JavaScript puro. Gráficos: Lightweight Charts (preço) e Chart.js (estatísticas).

## Como rodar
1. **(Uma vez) Baixe o dólar real**, para a udiCOIN ter âncora:
   ```
   python3 tools/baixar_dolar.py
   ```
   Baixa a série diária do dólar (Banco Central, 2010 → hoje) e gera `data/usdbrl.csv` e `web/data/usdbrl.js`.
   Sem acesso à API do BCB? Baixe o histórico em CSV (ex.: Yahoo Finance, símbolo `BRL=X`) e converta:
   ```
   python3 tools/baixar_dolar.py --csv arquivo.csv
   ```
2. Abra `web/index.html` no navegador (precisa de internet para Tailwind, fontes e bibliotecas de gráfico).
   Se algo não carregar: `python3 -m http.server 8000` dentro de `web/` e abra `http://localhost:8000`.

Sem o passo 1 o site funciona em **modo simulado** (sem âncora), e avisa isso no selo ao lado do título do gráfico.

## Como o preço é gerado

**Modo âncora** (quando `web/data/usdbrl.js` existe):

```
ln(preço udiCOIN) = ln(V) + d
d(t) = (1 − k)·d(t−1) + deriva do regime + ruído + impacto dos bots
```
- `V` = dólar real reescalado para começar em R$ 100 (o "valor justo"). **O agente nunca deve ver `V`.**
- `d` = desvio fictício, puxado de volta a zero por um elástico suave (`k = 0,02`, meia-vida ≈ 34 dias úteis).
- **Regimes escondidos** (alta, baixa, lateral) empurram o desvio.
- **Impacto realista dos bots:** o fluxo líquido de compras/vendas do dia vira um pequeno empurrão no preço do dia seguinte,
  e cada ordem paga slippage (meio spread + impacto proporcional ao tamanho da ordem). "Tamanho do mercado" = R$ 20 mi/dia (`IMPACT.depth`).
- Um passo = um **dia útil**; um ano = 252 passos. Se a série acabar, recomeça sem salto (e o selo avisa "série reiniciada").
- O botão **Âncora** do gráfico mostra `V` como linha pontilhada (só visualização).

**Modo simulado** (sem dados): regimes escondidos + ruído, sem dado real. Um passo = um dia.

> Ainda **não há eventos** (notícias, reuniões, baleias). Entram na próxima etapa.

## Ambiente em Python (agente de IA, passo 0)
No terminal, dentro da pasta `udicoin` (Windows / PowerShell):
```
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -q
python -m udicoin.market.paridade
```
- Se o PowerShell bloquear o `Activate.ps1`: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` e tente de novo.
- `pytest -q` roda os testes. Um deles (`test_paridade_js`) é pulado até existir o `data/usdbrl.csv` real.
- `python -m udicoin.market.paridade` compara o modelo em Python com os números que você mediu no console do site
  (usa o CSV real e imprime ✓ ou ✗ por medida).
- `python -m udicoin.env.demo` roda 3 políticas triviais no ambiente; `python -m udicoin.env.figuras` gera os gráficos em `docs/figuras/`. Especificação do MDP: `docs/passo0_ambiente.md`.
- Concluído até agora: **0.1** (dados), **0.2** (preço), **0.3** (conta com taxa e slippage), **0.4** (features) e **0.5** (ambiente `reset`/`step`). **0.7** (documento do MDP e gráficos). Próximo: passo 1 (baselines).

## Estrutura
```
tools/baixar_dolar.py    # baixa/converte o dólar → data/usdbrl.csv e web/data/usdbrl.js
data/usdbrl.csv          # (gerado) data,fechamento
src/udicoin/
├── data/loader.py       # 0.1 lê o CSV, valida, divide por data e sorteia janelas de episódio
├── market/
│   ├── price_model.py   # 0.2 o modelo de preço (dólar escondido × desvio, regimes, impacto)
│   ├── conta.py         # 0.3 caixa, posição, taxa, slippage, banda sem operar
│   ├── diagnostico.py   # medidas de quão solta/colada está a udiCOIN
│   └── paridade.py      # confere Python x site
└── env/
    ├── features.py      # 0.4 o que o agente enxerga (7 números públicos)
    ├── trading_env.py   # 0.5 o ambiente: reset(), step(ação) -> obs, recompensa
    └── demo.py          # 3 políticas triviais
tests/                   # pytest (dados sintéticos; só a paridade usa o CSV real)
requirements.txt  pyproject.toml
web/
├── index.html           # layout
├── css/style.css        # paleta e componentes
├── data/usdbrl.js       # (gerado) window.USDBRL = [[data, valor], ...]
└── js/
    ├── data-provider.js # provedor: preço (âncora/regimes) + bots de regra
    └── app.js           # interface; só conversa com o provedor
```

## Como ligar o motor Python depois
O `app.js` só usa a interface do provedor (documentada no topo de `data-provider.js`).
Crie um segundo provedor com **a mesma interface** (buscando dados de uma API) e troque a criação em `app.js`.
O agente de IA em Python lê `data/usdbrl.csv` direto, e o **ambiente** (não o agente) é quem usa a série:
o agente só enxerga o preço da udiCOIN e o histórico recente.

## Bots atuais
Bots de **regra** (cruzamento de médias 5/20 dias), com perfis agressivo/moderado/seguro (só parâmetros).
São a referência contra a qual o agente de IA será comparado. Ação inválida vira "manter".

## Atalhos
- `Espaço`: play/pause.
- Console do navegador: `udicoin.sim.advance(30)` avança 30 passos.
