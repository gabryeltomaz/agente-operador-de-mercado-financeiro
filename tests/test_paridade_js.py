"""Compara o modelo em Python com os números do site (JS). Só roda com o CSV REAL em data/usdbrl.csv."""
import pytest

from udicoin.data.loader import CAMINHO_PADRAO, carregar_dolar
from udicoin.market.paridade import comparar

pytestmark = pytest.mark.skipif(
    not CAMINHO_PADRAO.exists(), reason="precisa de data/usdbrl.csv (rode tools/baixar_dolar.py)"
)


def test_python_bate_com_o_site():
    linhas = comparar(carregar_dolar(), n_sementes=30)
    fora = [l for l in linhas if not l["ok"]]
    assert not fora, "fora da tolerância:\n" + "\n".join(
        f"  k={l['k']} {l['medida']}: python {l['python_media']:.2f} ± {l['python_dp']:.2f} vs site {l['js']:.2f} (tol {l['tolerancia']:.2f})"
        for l in fora
    )
