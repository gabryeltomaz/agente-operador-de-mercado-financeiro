import json
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from udicoin.agente.dqn import AgenteDQN
from udicoin.agente.exportar_dqn import exportar
from udicoin.env.features import calcular_features


def _ler(js):
    txt = js.read_text(encoding="utf-8")
    return json.loads(re.search(r"window\.AGENTE_DQN = (.*);\s*$", txt, re.S).group(1))


def _forward_numpy(camadas, x):
    a = np.asarray(x, float)
    for i, c in enumerate(camadas):
        a = np.asarray(c["W"]) @ a + np.asarray(c["b"])
        if i < len(camadas) - 1:
            a = np.maximum(a, 0)
    return a


def test_exporta_pesos_que_reproduzem_a_rede(tmp_path):
    ag = AgenteDQN(semente=3)
    pt = tmp_path / "rede.pt"
    torch.save(ag.rede.state_dict(), pt)
    d = _ler(exportar(pt, 5.0, tmp_path / "agente_dqn.js"))
    assert d["nivel"] == 5.0 and d["niveis"] == [0.0, 0.5, 1.0] and len(d["camadas"]) == 3
    assert len(d["testes"]) == 12
    for t in d["testes"]:
        obs = calcular_features(np.array(t["closes"]), t["exposicao"], t["lucro"])
        np.testing.assert_allclose(_forward_numpy(d["camadas"], obs), ag.notas(obs), atol=1e-4)
        np.testing.assert_allclose(_forward_numpy(d["camadas"], t["features"]), t["notas"], atol=1e-4)


def test_exporta_metadados_e_curva_quando_existe(tmp_path):
    ag = AgenteDQN(semente=1)
    pt = tmp_path / "dqn_5x_s0_final.pt"
    torch.save(ag.rede.state_dict(), pt)
    (tmp_path / "dqn_5x_s0_curva.csv").write_text(
        "episodio,epsilon,treino_log,val_pct,exposicao,comprar_segurar\n250,0.89,-0.085,6.28,0.53,0.5\n500,0.77,-0.053,15.47,0.41,0.5\n",
        encoding="utf-8")
    d = _ler(exportar(pt, 5.0, tmp_path / "agente_dqn.js"))
    assert [c["episodio"] for c in d["curva"]] == [250, 500] and d["treino"]["episodios"] == 500
    assert d["treino"]["gama"] == 0.99
    # sem CSV: curva vazia e episódios desconhecidos
    pt2 = tmp_path / "outra" ; pt2.mkdir()
    torch.save(ag.rede.state_dict(), pt2 / "dqn_5x_s0_final.pt")
    d2 = _ler(exportar(pt2 / "dqn_5x_s0_final.pt", 5.0, tmp_path / "b.js"))
    assert d2["curva"] == [] and d2["treino"]["episodios"] is None
