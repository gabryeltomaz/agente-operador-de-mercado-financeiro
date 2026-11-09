import numpy as np
import pytest

from udicoin.data.loader import CAMINHO_PADRAO
from udicoin.env.trading_env import ConfigAmbiente, UdiCoinEnv

pytestmark = pytest.mark.skipif(not CAMINHO_PADRAO.exists(), reason="precisa de data/usdbrl.csv")


@pytest.mark.parametrize("divisao", ["treino", "validacao", "teste"])
def test_episodios_completos_com_dolar_real(divisao):
    env = UdiCoinEnv.do_csv(ConfigAmbiente(divisao=divisao))
    for seed in range(3):
        obs, info = env.reset(seed=seed)
        fim, n = False, 0
        while not fim:
            obs, r, _, fim, info = env.step(n % 3)
            assert np.isfinite(obs).all() and np.isfinite(r)
            n += 1
        assert n == 252 and info["patrimonio"] > 0
