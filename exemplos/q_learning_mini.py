"""Q-learning no menor exemplo possível: um corredor com 4 casas.

    [0] [1] [2] [3=queijo]

O agente começa na casa 0. Ações: 0 = esquerda, 1 = direita.
Chegar na casa 3 dá recompensa +1 e acaba o episódio. Cada passo custa -0.01 (para ele ter pressa).
Rode:  python exemplos/q_learning_mini.py
"""
import numpy as np

N_ESTADOS, N_ACOES = 4, 2
ALFA = 0.5     # α: quanto eu aceito da informação nova (taxa de aprendizado)
GAMA = 0.9     # γ: quanto eu valorizo o futuro
EPSILON = 0.2  # ε: probabilidade de explorar (agir ao acaso)

rng = np.random.default_rng(0)
Q = np.zeros((N_ESTADOS, N_ACOES))  # a "tabela de notas": Q[estado, ação]


def passo(estado, acao):
    """O ambiente: devolve (novo_estado, recompensa, acabou)."""
    novo = max(0, estado - 1) if acao == 0 else estado + 1
    if novo == 3:
        return novo, 1.0, True
    return novo, -0.01, False


for episodio in range(1, 31):
    s, acabou = 0, False
    while not acabou:
        # 1) escolher: explora às vezes, senão usa a melhor nota
        a = int(rng.integers(N_ACOES)) if rng.random() < EPSILON else int(Q[s].argmax())
        # 2) agir e ver o que aconteceu
        s2, r, acabou = passo(s, a)
        # 3) aprender: aproximar Q[s,a] de  (recompensa + γ · melhor nota do próximo estado)
        alvo = r if acabou else r + GAMA * Q[s2].max()
        Q[s, a] += ALFA * (alvo - Q[s, a])
        s = s2
    if episodio in (1, 2, 5, 30):
        print(f"\nDepois do episódio {episodio}:  (colunas: esquerda, direita)")
        print(np.round(Q, 3))

print("\nPolítica aprendida (melhor ação em cada casa):")
for s in range(3):
    print(f"  casa {s}: {'direita' if Q[s].argmax() == 1 else 'esquerda'}")
