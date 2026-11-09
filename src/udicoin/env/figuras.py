"""Gera os gráficos do passo 0 em docs/figuras/:  python -m udicoin.env.figuras"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv

SAIDA = Path(__file__).resolve().parents[3] / "docs" / "figuras"


def episodio(env: UdiCoinEnv, politica, semente: int) -> dict:
    obs, info = env.reset(seed=semente)
    pat, exp, anc, preco, reg = [info["patrimonio"]], [info["exposicao"]], [], [info["fechamento"]], []
    anc.append(info["oculto"].ancora); reg.append(info["oculto"].regime)
    fim = False
    while not fim:
        obs, r, _, fim, info = env.step(politica(obs))
        pat.append(info["patrimonio"]); exp.append(info["exposicao"]); preco.append(info["fechamento"])
        anc.append(info["oculto"].ancora); reg.append(info["oculto"].regime)
    return {k: np.array(v) for k, v in dict(pat=pat, exp=exp, preco=preco, anc=anc, reg=reg).items()}


def gerar(env: UdiCoinEnv, saida: Path = SAIDA, semente: int = 4) -> list[Path]:
    saida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    arquivos = []

    # 1) preço x valor justo, com os regimes escondidos pintados ao fundo
    e = episodio(env, lambda o: 2, semente)
    fig, ax = plt.subplots(figsize=(9, 4))
    cores = {0: "#c8f0d0", 1: "#f6c9c9", 2: "#e6e6e6"}
    for i in range(len(e["reg"])):
        ax.axvspan(i - 0.5, i + 0.5, color=cores[int(e["reg"][i])], lw=0)
    ax.plot(e["preco"], label="preço da udiCOIN (o agente vê)", color="#1f4e9c")
    ax.plot(e["anc"], "--", label="valor justo V (escondido)", color="#444")
    ax.set(title="Preço x valor justo (fundo: verde=alta, vermelho=baixa, cinza=lateral)", xlabel="dia", ylabel="R$")
    ax.legend(); fig.tight_layout(); p = saida / "01_preco_vs_valor_justo.png"; fig.savefig(p, dpi=130); plt.close(fig); arquivos.append(p)

    # 2) distribuição do desvio ln(preço/V) em vários episódios
    d = np.concatenate([np.log(x["preco"] / x["anc"]) for x in (episodio(env, lambda o: 0, s) for s in range(30))]) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d, bins=50, color="#1f4e9c"); ax.axvline(0, color="k", lw=1)
    ax.set(title=f"Desvio do preço em relação ao valor justo (dp = {d.std():.1f}%)", xlabel="%", ylabel="dias")
    fig.tight_layout(); p = saida / "02_desvio_ancora.png"; fig.savefig(p, dpi=130); plt.close(fig); arquivos.append(p)

    # 3) um episódio: política aleatória x comprada e parada
    ea = episodio(env, lambda o: int(rng.integers(env.n_acoes)), semente)
    eb = episodio(env, lambda o: 2, semente)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(ea["pat"], label="aleatória"); a1.plot(eb["pat"], label="comprada e parada")
    a1.set(title="Mesmo episódio, duas políticas triviais", ylabel="patrimônio (R$)"); a1.legend()
    a2.step(range(len(ea["exp"])), ea["exp"], where="post", color="#c0392b"); a2.set(ylabel="exposição (aleatória)", xlabel="dia")
    fig.tight_layout(); p = saida / "03_episodio_politicas_triviais.png"; fig.savefig(p, dpi=130); plt.close(fig); arquivos.append(p)
    return arquivos


def main() -> None:
    for p in gerar(UdiCoinEnv.do_csv(ConfigAmbiente(divisao="treino"))):
        print("gerado:", p)


if __name__ == "__main__":
    main()
