"""Passo 2: o agente Q-learning tabular (o mesmo código do corredor, ligado na udiCOIN).

O ambiente devolve 7 números contínuos. A tabela Q precisa de estados "inteiros", então
DISCRETIZAMOS: transformamos alguns números em faixas.

    estado = (tendência curta 5d, tendência longa 20d, exposição atual)
             3 faixas           × 3 faixas            × 3 faixas        = 27 estados
    ações  = 0 (ficar fora), 1 (meio comprado), 2 (todo comprado)       → tabela Q de 27 × 3
"""
from __future__ import annotations

import numpy as np

IDX_RET_5D, IDX_RET_20D, IDX_EXPOSICAO = 1, 2, 5  # posições na observação (ver env/features.py)
LIMIAR_TENDENCIA = 0.5  # as features já estão em "desvios-padrão": abaixo de -0,5 = queda, acima de 0,5 = alta
N_ESTADOS = 27


def _faixa_tendencia(x: float) -> int:
    return 0 if x < -LIMIAR_TENDENCIA else (2 if x > LIMIAR_TENDENCIA else 1)  # 0 queda, 1 neutra, 2 alta


def _faixa_exposicao(x: float) -> int:
    return 0 if x < 0.25 else (1 if x < 0.75 else 2)  # 0 fora, 1 meio, 2 tudo


def discretizar(obs: np.ndarray) -> int:
    """Observação (7 números) -> índice do estado (0..26)."""
    curta = _faixa_tendencia(obs[IDX_RET_5D])
    longa = _faixa_tendencia(obs[IDX_RET_20D])
    expo = _faixa_exposicao(obs[IDX_EXPOSICAO])
    return (curta * 3 + longa) * 3 + expo


def descrever_estado(s: int) -> str:
    nomes_t, nomes_e = ("queda", "neutra", "alta"), ("fora", "meio", "comprado")
    curta, resto = divmod(s, 9)
    longa, expo = divmod(resto, 3)
    return f"curto prazo {nomes_t[curta]:6} | longo prazo {nomes_t[longa]:6} | estou {nomes_e[expo]}"


class AgenteQ:
    def __init__(self, n_acoes: int = 3, alfa: float = 0.1, gama: float = 0.9, semente: int = 0, n_estados: int = N_ESTADOS) -> None:
        self.n_acoes, self.alfa, self.gama = n_acoes, alfa, gama
        self.Q = np.zeros((n_estados, n_acoes))
        self.rng = np.random.default_rng(semente)

    def agir(self, s: int, epsilon: float) -> int:
        """ε-greedy: com prob. ε age ao acaso; senão pega a melhor nota (empate: sorteia)."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_acoes))
        melhores = np.flatnonzero(self.Q[s] == self.Q[s].max())
        return int(self.rng.choice(melhores))

    def aprender(self, s: int, a: int, r: float, s2: int) -> None:
        """A fórmula do passo 1. O episódio acaba por tempo (não por 'morte'), então sempre usamos o futuro."""
        alvo = r + self.gama * self.Q[s2].max()
        self.Q[s, a] += self.alfa * (alvo - self.Q[s, a])

    def politica(self) -> np.ndarray:
        """Melhor ação de cada estado (só para exibir; para decidir use agir(s, 0))."""
        return self.Q.argmax(axis=1)


def treinar(env, agente: AgenteQ, episodios: int = 300, eps_ini: float = 1.0, eps_fim: float = 0.05, semente: int = 0, disc=discretizar):
    """ε começa alto (explora muito) e cai linearmente até 70% do treino. Devolve o retorno de cada episódio."""
    retornos = []
    for ep in range(episodios):
        eps = max(eps_fim, eps_ini - (eps_ini - eps_fim) * ep / (0.7 * episodios))
        obs, info = env.reset(seed=semente * 100_000 + ep)
        s, fim, soma = disc(obs), False, 0.0
        while not fim:
            a = agente.agir(s, eps)
            obs, r, _, fim, info = env.step(a)
            s2 = disc(obs)
            agente.aprender(s, a, r, s2)
            s, soma = s2, soma + r
        retornos.append(soma)  # soma dos log-retornos = log(patrimônio final / inicial)
    return np.array(retornos)


def avaliar(env, escolher, seeds) -> np.ndarray:
    """Roda a política (função obs -> ação) em episódios novos e devolve o retorno % de cada um."""
    saida = []
    for sd in seeds:
        obs, info = env.reset(seed=sd)
        eq0, fim = info["patrimonio"], False
        while not fim:
            obs, r, _, fim, info = env.step(escolher(obs))
        saida.append(100 * (info["patrimonio"] / eq0 - 1))
    return np.array(saida)
