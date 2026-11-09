import numpy as np
import pandas as pd
import pytest

from udicoin.data.loader import (
    DadosInvalidos,
    Periodo,
    carregar_dolar,
    dividir,
    janela_ln,
    periodos,
    sortear_inicio,
)


def _csv(tmp_path, linhas, cabecalho="data,fechamento"):
    caminho = tmp_path / "x.csv"
    caminho.write_text(cabecalho + "\n" + "\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


def _linhas_validas(n=600):
    datas = pd.bdate_range("2012-01-02", periods=n).strftime("%Y-%m-%d")
    return [f"{d},{1.8 + 0.001 * i:.4f}" for i, d in enumerate(datas)]


def test_carrega_csv_valido(csv_sintetico):
    df = carregar_dolar(csv_sintetico)
    assert list(df.columns) == ["data", "fechamento"]
    assert df["data"].is_monotonic_increasing and (df["fechamento"] > 0).all()
    assert len(df) > 4000


def test_arquivo_inexistente_explica_o_que_fazer(tmp_path):
    with pytest.raises(FileNotFoundError, match="baixar_dolar"):
        carregar_dolar(tmp_path / "nao_existe.csv")


@pytest.mark.parametrize(
    "linhas, cabecalho, trecho",
    [
        (_linhas_validas(), "data,valor", "faltam as colunas"),
        (_linhas_validas()[:100], "data,fechamento", "poucos dados"),
        (_linhas_validas() + [_linhas_validas()[10]], "data,fechamento", "repetidas"),
        (list(reversed(_linhas_validas())), "data,fechamento", "ordem crescente"),
        (["2012-01-02,abc"] + _linhas_validas()[1:], "data,fechamento", "não numéricos"),
        (["2012-01-02,-1.0"] + _linhas_validas()[1:], "data,fechamento", "menores ou iguais a zero"),
        (["02/01/2012,1.8"] + _linhas_validas()[1:], "data,fechamento", "AAAA-MM-DD"),
    ],
)
def test_rejeita_dados_ruins(tmp_path, linhas, cabecalho, trecho):
    with pytest.raises(DadosInvalidos, match=trecho):
        carregar_dolar(_csv(tmp_path, linhas, cabecalho))


def test_avisa_de_saltos_grandes(tmp_path):
    linhas = _linhas_validas()
    linhas[300] = linhas[300].split(",")[0] + ",3.5"  # salto absurdo
    df = carregar_dolar(_csv(tmp_path, linhas))
    assert any("salto" in a for a in df.attrs["avisos"])


def test_divisoes_cobrem_tudo_sem_se_misturar(csv_sintetico):
    df = carregar_dolar(csv_sintetico)
    p = periodos(df)
    assert p["treino"].datas.max() < p["validacao"].datas.min()
    assert p["validacao"].datas.max() < p["teste"].datas.min()
    assert p["treino"].datas.min() >= np.datetime64("2010-01-01")
    assert p["treino"].datas.max() <= np.datetime64("2019-12-31")
    assert p["validacao"].datas.min() >= np.datetime64("2020-01-01")
    assert p["validacao"].datas.max() <= np.datetime64("2022-12-31")
    assert p["teste"].datas.min() >= np.datetime64("2023-01-01")
    assert sum(len(x) for x in p.values()) == len(df)


def test_divisao_desconhecida(csv_sintetico):
    with pytest.raises(KeyError):
        dividir(carregar_dolar(csv_sintetico), "producao")


def test_janelas_ficam_dentro_do_periodo(csv_sintetico):
    p = periodos(carregar_dolar(csv_sintetico))["teste"]
    rng = np.random.default_rng(0)
    passos = 252
    inicios = [sortear_inicio(p, passos, rng) for _ in range(500)]
    assert min(inicios) >= 0 and max(inicios) + passos + 1 <= len(p)
    for i in inicios[:20]:
        assert len(janela_ln(p, i, passos)) == passos + 1


def test_periodo_curto_demais_para_um_episodio():
    curto = Periodo("teste", np.arange(100).astype("datetime64[D]"), np.linspace(1.0, 2.0, 100))
    with pytest.raises(DadosInvalidos, match="precisa de 253"):
        sortear_inicio(curto, 252, np.random.default_rng(0))


def test_sorteio_e_reproduzivel(csv_sintetico):
    p = periodos(carregar_dolar(csv_sintetico))["treino"]
    a = [sortear_inicio(p, 252, np.random.default_rng(5)) for _ in range(3)]
    assert a == [sortear_inicio(p, 252, np.random.default_rng(5)) for _ in range(3)]
