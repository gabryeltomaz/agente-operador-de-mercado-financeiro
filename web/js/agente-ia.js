/**
 * Agente de IA (DQN) no navegador: só a conta da rede neural, sem biblioteca.
 * Os pesos vêm de data/agente_dqn.js (gerado por `python -m udicoin.agente.exportar_dqn`).
 *
 * É a MESMA conta do Python:
 *   observação (7 números) -> Linear(7,64) -> ReLU -> Linear(64,64) -> ReLU -> Linear(64,3) -> 3 notas
 *   ação = a de maior nota; vira uma exposição-alvo (0, 0,5 ou 1).
 * As features são idênticas às de src/udicoin/env/features.py.
 * Interface: AgenteIA.pronto()  AgenteIA.decidir(closes, exposicao, lucro)  AgenteIA.autoteste()
 */
(function (global) {
  'use strict';

  const LOOKBACK = 20; // precisa de 21 fechamentos
  const SIGMA_REF = 0.008;
  const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

  const pesos = () => global.AGENTE_DQN || null;
  const pronto = () => { const p = pesos(); return !!(p && Array.isArray(p.camadas) && p.camadas.length > 0); };

  /** closes: os últimos fechamentos (usa os 21 mais recentes). Devolve as 7 features. */
  function features(closes, exposicao, lucro) {
    if (closes.length < LOOKBACK + 1) throw new Error('precisa de ' + (LOOKBACK + 1) + ' fechamentos');
    const c = closes.slice(-(LOOKBACK + 1));
    const ln = c.map(Math.log);
    const r = [];
    for (let i = 1; i < ln.length; i++) r.push(ln[i] - ln[i - 1]);
    const last = ln.length - 1;
    const media = (v) => v.reduce((s, x) => s + x, 0) / v.length;
    const m = media(r);
    const dp = Math.sqrt(media(r.map((x) => (x - m) * (x - m)))); // desvio padrão populacional (como o numpy)
    return [
      clip((ln[last] - ln[last - 1]) / SIGMA_REF, -3, 3),
      clip((ln[last] - ln[last - 5]) / (SIGMA_REF * Math.sqrt(5)), -3, 3),
      clip((ln[last] - ln[0]) / (SIGMA_REF * Math.sqrt(LOOKBACK)), -3, 3),
      clip((media(c.slice(-5)) / media(c) - 1) / 0.01, -3, 3),
      clip(dp / SIGMA_REF, 0, 3),
      clip(exposicao, 0, 1),
      clip(lucro / 0.05, -3, 3),
    ];
  }

  /** A rede: camadas lineares com ReLU entre elas. */
  function forward(x) {
    const camadas = pesos().camadas;
    let a = x;
    camadas.forEach((L, i) => {
      const out = L.b.map((b, j) => {
        const w = L.W[j];
        let s = b;
        for (let k = 0; k < w.length; k++) s += w[k] * a[k];
        return s;
      });
      a = i < camadas.length - 1 ? out.map((v) => (v > 0 ? v : 0)) : out;
    });
    return a;
  }

  /** Ativação de TODAS as camadas para uma observação: [entrada, oculta1, oculta2, saída]. Usado pela tela "Rede neural". */
  function ativacoes(x) {
    const camadas = pesos().camadas;
    const todas = [x.slice()];
    let a = x;
    camadas.forEach((L, i) => {
      const out = L.b.map((b, j) => {
        const w = L.W[j];
        let s = b;
        for (let k = 0; k < w.length; k++) s += w[k] * a[k];
        return s;
      });
      a = i < camadas.length - 1 ? out.map((v) => (v > 0 ? v : 0)) : out;
      todas.push(a);
    });
    return todas;
  }

  /** Tamanhos das camadas e número de parâmetros (pesos + vieses). */
  function info() {
    const c = pesos().camadas;
    const tamanhos = [c[0].W[0].length].concat(c.map((L) => L.b.length));
    const parametros = c.reduce((s, L) => s + L.b.length * L.W[0].length + L.b.length, 0);
    return { tamanhos, parametros };
  }

  function decidir(closes, exposicao, lucro) {
    const f = features(closes, exposicao, lucro);
    const notas = forward(f);
    let acao = 0;
    for (let i = 1; i < notas.length; i++) if (notas[i] > notas[acao]) acao = i; // empate: a primeira (como o argmax)
    return { acao, alvo: pesos().niveis[acao], notas, features: f };
  }

  /** Confere o JavaScript contra os vetores calculados no PyTorch. Rode no console: AgenteIA.autoteste() */
  function autoteste() {
    if (!pronto()) return { ok: false, erro: 'pesos não carregados' };
    let maxF = 0, maxN = 0, acoesIguais = true;
    for (const t of pesos().testes || []) {
      const f = features(t.closes, t.exposicao, t.lucro);
      f.forEach((v, i) => { maxF = Math.max(maxF, Math.abs(v - t.features[i])); });
      const n = forward(t.features); // usa as features do Python: isola só a rede
      n.forEach((v, i) => { maxN = Math.max(maxN, Math.abs(v - t.notas[i])); });
      const am = (v) => v.indexOf(Math.max(...v));
      if (am(n) !== am(t.notas)) acoesIguais = false;
    }
    return { ok: maxF < 1e-4 && maxN < 1e-3 && acoesIguais, dif_features: maxF, dif_notas: maxN, mesma_acao: acoesIguais };
  }

  global.AgenteIA = { LOOKBACK, pronto, features, forward, ativacoes, info, decidir, autoteste, pesos, nivel: () => (pesos() ? pesos().nivel : null) };
})(typeof window !== 'undefined' ? window : globalThis);
