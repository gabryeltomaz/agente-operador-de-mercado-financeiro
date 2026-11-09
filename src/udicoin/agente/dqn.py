"""O agente DQN (Deep Q-Network): a tabela Q substituída por uma rede neural.

Na tabela:  nota = Q[estado, ação]          (estado = uma das 27 faixas)
No DQN:     notas = rede(observação)        (observação = os 7 números contínuos; devolve 1 nota por ação)

A fórmula de aprendizado é a MESMA do Q-learning:   alvo = r + γ · max Q(s', ·)
Só que em vez de "aproximar uma célula da tabela", ajustamos os pesos da rede para a saída chegar perto do alvo.

Três peças novas, cada uma resolvendo um problema que a rede traz:
  1. RedeQ      a rede em si (7 entradas -> 64 -> 64 -> 3 saídas)
  2. Memoria    "replay buffer": guarda jogadas passadas e as sorteia ao acaso no treino
  3. rede alvo  uma cópia congelada da rede, atualizada de vez em quando, para o alvo não se mexer a cada passo
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)  # a rede é pequena: 1 thread é mais rápido que várias


class RedeQ(nn.Module):
    """Peça 1: observação (7 números) -> nota de cada ação."""

    def __init__(self, dim_obs: int, n_acoes: int, ocultos: int = 64) -> None:
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Linear(dim_obs, ocultos), nn.ReLU(),
            nn.Linear(ocultos, ocultos), nn.ReLU(),
            nn.Linear(ocultos, n_acoes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.camadas(x)


class Memoria:
    """Peça 2: replay buffer circular. Guarda (s, a, r, s2) e sorteia lotes aleatórios.

    Por quê? Dias seguidos são muito parecidos entre si. Aprender na ordem em que acontecem deixa a rede
    "viciada" no momento atual e faz ela esquecer o resto. Sortear jogadas antigas e novas misturadas ajuda.
    """

    def __init__(self, capacidade: int, dim_obs: int, semente: int = 0) -> None:
        self.cap, self.n, self.i = capacidade, 0, 0
        self.s = np.zeros((capacidade, dim_obs), np.float32)
        self.s2 = np.zeros((capacidade, dim_obs), np.float32)
        self.a = np.zeros(capacidade, np.int64)
        self.r = np.zeros(capacidade, np.float32)
        self.rng = np.random.default_rng(semente)

    def guardar(self, s, a, r, s2) -> None:
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i] = s, a, r, s2
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sortear(self, tamanho: int):
        idx = self.rng.integers(0, self.n, tamanho)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]))

    def __len__(self) -> int:
        return self.n


class AgenteDQN:
    def __init__(self, dim_obs: int = 7, n_acoes: int = 3, gama: float = 0.99, lr: float = 1e-3,
                 lote: int = 64, capacidade: int = 50_000, atualizar_alvo: int = 500, minimo_memoria: int = 1000,
                 treinar_a_cada: int = 1, semente: int = 0) -> None:
        torch.manual_seed(semente)
        self.n_acoes, self.gama, self.lote = n_acoes, gama, lote
        self.atualizar_alvo, self.minimo_memoria, self.treinar_a_cada = atualizar_alvo, minimo_memoria, treinar_a_cada
        self.rede = RedeQ(dim_obs, n_acoes)                 # a que aprende
        self.rede_alvo = RedeQ(dim_obs, n_acoes)            # peça 3: a cópia congelada
        self.rede_alvo.load_state_dict(self.rede.state_dict())
        self.otimizador = torch.optim.Adam(self.rede.parameters(), lr=lr)
        self.memoria = Memoria(capacidade, dim_obs, semente)
        self.rng = np.random.default_rng(semente)
        self.passos = 0

    def agir(self, obs: np.ndarray, epsilon: float) -> int:
        """ε-greedy, igual à tabela: às vezes explora, senão pega a ação de maior nota da rede."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_acoes))
        with torch.no_grad():
            return int(self.rede(torch.from_numpy(np.asarray(obs, np.float32))).argmax())

    def notas(self, obs: np.ndarray) -> np.ndarray:
        """As notas que a rede dá a cada ação (para inspecionar/mostrar)."""
        with torch.no_grad():
            return self.rede(torch.from_numpy(np.asarray(obs, np.float32))).numpy()

    def aprender(self, s, a, r, s2) -> None:
        """Guarda a jogada e, se já há memória suficiente, dá UM passo de aprendizado com um lote sorteado."""
        self.memoria.guardar(s, a, r, s2)
        self.passos += 1
        if len(self.memoria) < self.minimo_memoria or self.passos % self.treinar_a_cada != 0:
            return
        S, A, R, S2 = self.memoria.sortear(self.lote)
        nota_atual = self.rede(S).gather(1, A.unsqueeze(1)).squeeze(1)  # a nota que a rede deu à ação tomada
        with torch.no_grad():
            alvo = R + self.gama * self.rede_alvo(S2).max(dim=1).values  # r + γ · melhor nota do próximo (rede congelada)
        perda = nn.functional.smooth_l1_loss(nota_atual, alvo)           # quão longe a nota ficou do alvo
        self.otimizador.zero_grad()
        perda.backward()
        nn.utils.clip_grad_norm_(self.rede.parameters(), 10.0)
        self.otimizador.step()
        if self.passos % self.atualizar_alvo == 0:                       # de vez em quando, copia a rede para a rede alvo
            self.rede_alvo.load_state_dict(self.rede.state_dict())


def treinar_dqn(env, agente: AgenteDQN, episodios: int = 300, eps_ini: float = 1.0, eps_fim: float = 0.05, semente: int = 0):
    """Mesmo laço do Q-learning: ε cai linearmente até 70% do treino. Devolve o retorno (log) de cada episódio."""
    retornos = []
    for ep in range(episodios):
        eps = max(eps_fim, eps_ini - (eps_ini - eps_fim) * ep / (0.7 * episodios))
        obs, info = env.reset(seed=semente * 100_000 + ep)
        fim, soma = False, 0.0
        while not fim:
            a = agente.agir(obs, eps)
            obs2, r, _, fim, info = env.step(a)
            agente.aprender(obs, a, r, obs2)
            obs, soma = obs2, soma + r
        retornos.append(soma)
    return np.array(retornos)


def medir_dqn(env, agente: AgenteDQN, seeds):
    """Retorno % médio e exposição média, com ε = 0, em episódios novos."""
    rets, exps = [], []
    for sd in seeds:
        obs, info = env.reset(seed=sd)
        eq0, fim, e = info["patrimonio"], False, []
        while not fim:
            obs, r, _, fim, info = env.step(agente.agir(obs, 0.0))
            e.append(info["exposicao"])
        rets.append(100 * (info["patrimonio"] / eq0 - 1)); exps.append(np.mean(e))
    return float(np.mean(rets)), float(np.mean(exps))
