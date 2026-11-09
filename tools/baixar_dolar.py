#!/usr/bin/env python3
"""Baixa a série diária do dólar (2010 → hoje) e gera os arquivos que o projeto usa.

Fonte: Banco Central do Brasil, SGS série 1 (dólar americano, venda, diário).

Saídas (relativas à pasta do projeto, que contém tools/):
  data/usdbrl.csv       data,fechamento           (para análise, notebooks e o agente em Python)
  web/data/usdbrl.js    window.USDBRL = [[data, valor], ...]   (o site lê este arquivo)

Uso:
  python3 tools/baixar_dolar.py                  # baixa do BCB (precisa de internet)
  python3 tools/baixar_dolar.py --desde 2010-01-01
  python3 tools/baixar_dolar.py --csv meu_arquivo.csv   # converte um CSV baixado à mão
                                                        # (Yahoo Finance, Investing, etc.)
Só usa a biblioteca padrão do Python.
"""
import argparse
import csv
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERIE_BCB = 1  # dólar americano (venda), diário
URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?formato=json&dataInicial={ini}&dataFinal={fim}"
RAIZ = Path(__file__).resolve().parent.parent
FONTE_BCB = "Banco Central do Brasil, SGS série 1 (dólar americano, venda, diário)"


def _br(d: dt.date) -> str:
    return d.strftime("%d/%m/%Y")


def buscar_bcb(ini: dt.date, fim: dt.date) -> list:
    """Uma requisição ao BCB. Separada para facilitar testes."""
    url = URL.format(serie=SERIE_BCB, ini=_br(ini), fim=_br(fim))
    req = urllib.request.Request(url, headers={"User-Agent": "udicoin-downloader"})
    ultimo_erro = None
    for tentativa in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            ultimo_erro = e
            time.sleep(1.5 * (tentativa + 1))
    raise RuntimeError(f"Falha ao consultar o Banco Central: {ultimo_erro}")


def janelas(ini: dt.date, fim: dt.date, anos: int = 5):
    """A API limita o tamanho da janela de séries diárias; pedimos em pedaços de 5 anos."""
    atual = ini
    while atual <= fim:
        prox = min(dt.date(atual.year + anos, atual.month, min(atual.day, 28)) - dt.timedelta(days=1), fim)
        yield atual, prox
        atual = prox + dt.timedelta(days=1)


def baixar(ini: dt.date, fim: dt.date, buscar=buscar_bcb) -> list:
    linhas = {}
    for a, b in janelas(ini, fim):
        for item in buscar(a, b):
            d = dt.datetime.strptime(item["data"], "%d/%m/%Y").date().isoformat()
            v = float(str(item["valor"]).replace(",", "."))
            if v > 0:
                linhas[d] = v
    return sorted(linhas.items())


def _achar(cabecalho, nomes):
    baixo = [c.strip().lower() for c in cabecalho]
    for n in nomes:
        if n in baixo:
            return baixo.index(n)
    return None


def _data(s: str) -> str:
    s = s.strip().strip('"')
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y"):
        try:
            return dt.datetime.strptime(s[:10], fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"data não reconhecida: {s!r}")


def ler_csv(caminho: Path) -> list:
    """Aceita CSV com coluna de data (data/date) e de preço (fechamento/close/adj close/valor/último)."""
    texto = caminho.read_text(encoding="utf-8-sig")
    delim = ";" if texto.count(";") > texto.count(",") else ","
    linhas = list(csv.reader(texto.splitlines(), delimiter=delim))
    cab = linhas[0]
    i_data = _achar(cab, ["data", "date"])
    i_val = _achar(cab, ["fechamento", "close", "adj close", "valor", "último", "ultimo", "price"])
    if i_data is None or i_val is None:
        raise SystemExit(f"Não achei as colunas de data e preço em {cab}")
    saida = {}
    for l in linhas[1:]:
        if len(l) <= max(i_data, i_val) or not l[i_val].strip() or l[i_val].strip().lower() == "null":
            continue
        bruto = l[i_val].strip().replace(" ", "")
        # "5,1234" (BR) ou "5.1234" (US); "1.234,56" -> 1234.56
        if "," in bruto and "." in bruto:
            bruto = bruto.replace(".", "").replace(",", ".")
        else:
            bruto = bruto.replace(",", ".")
        v = float(bruto)
        if v > 0:
            saida[_data(l[i_data])] = v
    return sorted(saida.items())


def validar(serie: list) -> None:
    if len(serie) < 500:
        raise SystemExit(f"Poucos dados ({len(serie)} dias). Algo deu errado.")
    for (d0, v0), (d1, v1) in zip(serie, serie[1:]):
        if abs(v1 / v0 - 1) > 0.15:
            print(f"  aviso: salto de {abs(v1 / v0 - 1):.0%} entre {d0} e {d1}", file=sys.stderr)


def salvar(serie: list, fonte: str) -> None:
    (RAIZ / "data").mkdir(exist_ok=True)
    (RAIZ / "web" / "data").mkdir(parents=True, exist_ok=True)
    with open(RAIZ / "data" / "usdbrl.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["data", "fechamento"])
        w.writerows(serie)
    meta = {"fonte": fonte, "de": serie[0][0], "ate": serie[-1][0], "dias": len(serie),
            "gerado_em": dt.date.today().isoformat()}
    pares = ",".join(f'["{d}",{v}]' for d, v in serie)
    js = f"// Gerado por tools/baixar_dolar.py. Não edite.\nwindow.USDBRL_META = {json.dumps(meta, ensure_ascii=False)};\nwindow.USDBRL = [{pares}];\n"
    (RAIZ / "web" / "data" / "usdbrl.js").write_text(js, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desde", default="2010-01-01", help="data inicial (AAAA-MM-DD)")
    ap.add_argument("--ate", default=dt.date.today().isoformat(), help="data final (AAAA-MM-DD)")
    ap.add_argument("--csv", help="converte um CSV baixado à mão em vez de consultar o BCB")
    a = ap.parse_args()
    ini, fim = dt.date.fromisoformat(a.desde), dt.date.fromisoformat(a.ate)

    if a.csv:
        serie = [(d, v) for d, v in ler_csv(Path(a.csv)) if a.desde <= d <= a.ate]
        fonte = f"CSV fornecido ({Path(a.csv).name})"
    else:
        print(f"Baixando dólar (BCB) de {ini} a {fim}...")
        try:
            serie = baixar(ini, fim)
        except RuntimeError as e:
            raise SystemExit(
                f"{e}\nSem acesso à API do BCB? Baixe o histórico do dólar em CSV (ex.: Yahoo Finance, símbolo BRL=X)"
                " e rode:\n  python3 tools/baixar_dolar.py --csv arquivo.csv"
            )
        fonte = FONTE_BCB
    validar(serie)
    salvar(serie, fonte)
    print(f"OK: {len(serie)} dias, de {serie[0][0]} a {serie[-1][0]}")
    print(f"  {RAIZ / 'data' / 'usdbrl.csv'}")
    print(f"  {RAIZ / 'web' / 'data' / 'usdbrl.js'}")


if __name__ == "__main__":
    main()
