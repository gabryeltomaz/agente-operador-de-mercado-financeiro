"""Dados SINTÉTICOS só para os testes. Nada aqui é o dólar real."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def ln_sintetico() -> np.ndarray:
    """Passeio aleatório diário (vol ~0,75%) fazendo o papel do log do dólar."""
    rng = np.random.default_rng(123)
    return np.log(1.75) + np.cumsum(rng.normal(0.0002, 0.0075, 4000))


@pytest.fixture
def csv_sintetico(tmp_path):
    """CSV no formato do projeto (data,fechamento), dias úteis de 2010 a 2026."""
    datas = pd.bdate_range("2010-01-04", "2026-10-29")
    rng = np.random.default_rng(1)
    valores = 1.75 * np.exp(np.cumsum(rng.normal(0.0002, 0.0075, len(datas))))
    caminho = tmp_path / "usdbrl.csv"
    pd.DataFrame({"data": datas.strftime("%Y-%m-%d"), "fechamento": np.round(valores, 4)}).to_csv(
        caminho, index=False
    )
    return caminho


@pytest.fixture(scope="session")
def periodos_sint():
    """Períodos treino/validação/teste a partir de um dólar sintético."""
    from udicoin.data.loader import carregar_dolar, periodos
    import tempfile, pathlib
    datas = pd.bdate_range("2010-01-04", "2026-10-29")
    rng = np.random.default_rng(1)
    valores = 1.75 * np.exp(np.cumsum(rng.normal(0.0002, 0.0075, len(datas))))
    d = pathlib.Path(tempfile.mkdtemp()) / "usdbrl.csv"
    pd.DataFrame({"data": datas.strftime("%Y-%m-%d"), "fechamento": np.round(valores, 4)}).to_csv(d, index=False)
    return periodos(carregar_dolar(d))
