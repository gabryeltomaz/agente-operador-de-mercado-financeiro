/**
 * udiCOIN — provedor de dados SIMULADO (mock).
 *
 * O front só conversa com este objeto. Quando o motor Python existir, basta
 * criar outro provedor com a MESMA interface (ex.: falando com uma API) e
 * trocar a linha `UdiProvider.create(...)` em app.js. O resto do site não muda.
 *
 * ── MODOS DE PREÇO ───────────────────────────────────────────────────────
 * 'anchor'  : udiCOIN = dólar real (reescalado p/ começar em R$ 100) × desvio fictício.
 *             ln(preço) = ln(V) + d, com  d(t) = (1-k)·d(t-1) + regime + ruído + impacto dos bots.
 *             V é o "valor justo" (a série real). O agente NUNCA deve vê-lo.
 *             Um passo = um dia útil (a série só tem dias úteis).
 * 'regimes' : sem dado real; regimes escondidos (alta/baixa/lateral) + ruído. Um passo = um dia.
 *
 * ── CONTRATO DE DADOS ────────────────────────────────────────────────────
 * candle : { dia, data:'YYYY-MM-DD', open, high, low, close, volume, regime,
 *            ancora?, ancora_data? }   (ancora* só no modo 'anchor', só p/ visualização)
 * op     : { id, dia, data, botId, botNome, tipo:'compra'|'venda',
 *            quantidade, preco, taxa, motivo }   (preco = preço executado, com slippage)
 * bot    : { id, nome, perfil, perfil_label, capital_inicial, saldo, caixa,
 *            unidades, posicao, lucro_pct, operacoes }
 * ponto  : { dia, data, equity, pos, units, ops, lucro_pct }   (série do bot)
 *
 * ── INTERFACE ────────────────────────────────────────────────────────────
 * create({ seed, mode, anchorData:[[ 'YYYY-MM-DD', valor ], ...], anchorMeta,
 *          anchorRange:{from,to}, anchorStart:'random'|'first'|'YYYY-MM-DD', k, depth })
 * on('update', fn)  fn({ candles:[novos], ops:[novas], bots:[...], state })
 * on('state', fn)   fn({ playing, speed, dia, data, preco, loops, modo })
 * play() pause() toggle() setSpeed(x) advance(dias)
 * addBot({ nome, perfil, capital })  -> bot   (lança Error se inválido)
 * getCandles() getBots() getOperations(limite) getSeries(botId)
 * getTotalSeries(n) getOpsCount() getState() getConfig()
 */
(function (global) {
  'use strict';

  const START = Date.UTC(2025, 0, 1);
  const MON0 = Date.UTC(2024, 11, 30); // uma segunda-feira: base do calendário de dias úteis
  const DAY_MS = 86400000;
  const FEE = 0.001; // 0,1% por operação
  const P0 = 100; // preço inicial da udiCOIN
  const MAX_ADVANCE = 3650;

  /** Perfis: só parâmetros. Depois viram "pesos da recompensa" no agente de IA. */
  const PROFILES = {
    agressivo: {
      label: 'Agressivo',
      fracBuy: 0.9, fracSell: 1, thr: 0.004, impulse: 0.04, stop: 0.12,
      desc: 'Entra forte (90% do caixa) e reage a sinais pequenos. Ganha mais, mas cai mais.',
    },
    moderado: {
      label: 'Moderado',
      fracBuy: 0.5, fracSell: 0.7, thr: 0.01, impulse: 0.01, stop: 0.08,
      desc: 'Meio-termo: posições médias e só age quando a tendência está mais firme.',
    },
    seguro: {
      label: 'Seguro',
      fracBuy: 0.25, fracSell: 1, thr: 0.02, impulse: 0, stop: 0.04,
      desc: 'Posições pequenas, só age em sinais claros e corta perdas cedo.',
    },
    ia: {
      label: 'IA (DQN)',
      desc: 'Rede neural treinada por reforço (DQN). Lê o histórico de preço e a própria conta e escolhe a exposição: 0%, 50% ou 100%.',
    },
  };
  const IA_BANDA = 0.02; // como no Python: não opera se a exposição já está a menos de 2 pontos do alvo

  /** Regimes escondidos (modo 'regimes'): é o padrão que a IA poderá descobrir. */
  const REGIMES = {
    alta: { mu: 0.002, sigma: 0.012 },
    baixa: { mu: -0.0022, sigma: 0.016 },
    lateral: { mu: 0, sigma: 0.008 },
  };
  const TRANS = {
    alta: [['alta', 0.965], ['lateral', 0.025], ['baixa', 0.01]],
    lateral: [['lateral', 0.94], ['alta', 0.035], ['baixa', 0.025]],
    baixa: [['baixa', 0.955], ['lateral', 0.035], ['alta', 0.01]],
  };

  /**
   * Modo 'anchor'. O regime empurra o DESVIO em relação ao dólar (não o preço todo).
   * k = força do elástico (suave: meia-vida ≈ 34 dias). Desvio típico fica em poucos %.
   */
  const ANCHOR = {
    k: 0.02,
    mu: { alta: 0.0012, baixa: -0.0012, lateral: 0 },
    sigma: { alta: 0.004, baixa: 0.0052, lateral: 0.0032 },
    stepsPerYear: 252,
    refSigma: 0.008, // volatilidade diária de referência (p/ volume e pavios)
  };
  /** Impacto realista: depth = "tamanho do mercado" em R$/dia. Quanto maior, menor o impacto. */
  const IMPACT = { lambda: 1, depth: 2e7, half: 0.0002 };

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const r2 = (x) => Math.round(x * 100) / 100;
  const r4 = (x) => Math.round(x * 10000) / 10000;
  const isoCal = (dia) => new Date(START + dia * DAY_MS).toISOString().slice(0, 10);
  /** n-ésimo dia útil a partir de 01/01/2025 (pula sábados e domingos). */
  const isoBiz = (n) => {
    const idx = n + 2; // 01/01/2025 é quarta-feira (segunda = 0)
    return new Date(MON0 + (Math.floor(idx / 5) * 7 + (idx % 5)) * DAY_MS).toISOString().slice(0, 10);
  };

  class MockProvider {
    constructor(opts = {}) {
      const seed = opts.seed != null ? opts.seed : 42;
      this.rng = mulberry32(seed); // só o preço usa este
      this.botRng = mulberry32(seed + 1000); // personalidade e impulsos dos bots
      this.anchorRng = mulberry32(seed + 2000); // trecho inicial da série real
      this.mode = opts.mode === 'anchor' ? 'anchor' : 'regimes';
      this.k = 0;
      this.trendMult = opts.trendMult > 0 ? Number(opts.trendMult) : 1; // força das tendências (regimes) no modo âncora
      this.anchor = null;
      this.impact = { lambda: 0, depth: 1, half: 0 }; // modo 'regimes': sem impacto nem spread
      this._pendingFlow = 0; // fluxo líquido (R$) dos bots no dia; vira impacto no dia seguinte
      this.candles = [];
      this.ops = [];
      this.bots = [];
      this.regime = 'lateral';
      this.listeners = { update: [], state: [] };
      this.playing = false;
      this.speed = 1;
      this.timer = null;
      this._opId = 0;
      this._botId = 0;

      if (this.mode === 'anchor') this._initAnchor(opts);

      const initial = opts.initialBots || [
        { nome: 'Alfa', perfil: 'agressivo', capital: 10000 },
        { nome: 'Beta', perfil: 'moderado', capital: 10000 },
        { nome: 'Gama', perfil: 'seguro', capital: 10000 },
        { nome: 'Delta', perfil: 'agressivo', capital: 10000 },
      ];
      if (!opts.initialBots && global.AgenteIA && global.AgenteIA.pronto()) initial.push({ nome: 'Rede', perfil: 'ia', capital: 10000 });
      initial.forEach((b) => this._createBot(b));
      const warm = opts.warmupDays != null ? opts.warmupDays : 120;
      for (let i = 0; i < warm; i++) this._step();
    }

    _initAnchor(opts) {
      const raw = Array.isArray(opts.anchorData) ? opts.anchorData : [];
      const from = (opts.anchorRange && opts.anchorRange.from) || '0000-00-00';
      const to = (opts.anchorRange && opts.anchorRange.to) || '9999-99-99';
      const byDate = new Map();
      for (const r of raw) {
        const data = String(r[0]); const v = Number(r[1]);
        if (data >= from && data <= to && isFinite(v) && v > 0) byDate.set(data, v);
      }
      const rows = Array.from(byDate, ([data, v]) => ({ data, v })).sort((a, b) => (a.data < b.data ? -1 : 1));
      if (rows.length < 60) throw new Error('Âncora: poucos dados (' + rows.length + ' dias no intervalo).');
      let start = 0;
      const s = opts.anchorStart;
      if (s === 'first') start = 0;
      else if (typeof s === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(s)) {
        start = Math.max(0, rows.findIndex((r) => r.data >= s));
      } else start = Math.floor(this.anchorRng() * rows.length);
      this.anchor = {
        rows, ln: rows.map((r) => Math.log(r.v)), i: start,
        lnV: Math.log(P0), d: 0, loops: 0, meta: opts.anchorMeta || null,
      };
      this.k = opts.k != null ? opts.k : ANCHOR.k;
      this.impact = { lambda: IMPACT.lambda, depth: opts.depth != null ? opts.depth : IMPACT.depth, half: IMPACT.half };
    }

    _date(dia) { return this.mode === 'anchor' ? isoBiz(dia) : isoCal(dia); }

    /* ── eventos ─────────────────────────────────────────── */
    on(evt, fn) {
      (this.listeners[evt] = this.listeners[evt] || []).push(fn);
      return () => { this.listeners[evt] = this.listeners[evt].filter((f) => f !== fn); };
    }
    _emit(evt, payload) { (this.listeners[evt] || []).forEach((f) => f(payload)); }
    _emitUpdate(candles, ops) {
      this._emit('update', { candles, ops, bots: this.getBots(), state: this.getState() });
    }

    /* ── controle da simulação ───────────────────────────── */
    getConfig() {
      const a = this.anchor;
      return {
        mode: this.mode,
        stepsPerYear: this.mode === 'anchor' ? ANCHOR.stepsPerYear : 365,
        k: this.k, depth: this.impact.depth, trendMult: this.trendMult,
        anchor: a ? { from: a.rows[0].data, to: a.rows[a.rows.length - 1].data, rows: a.rows.length, meta: a.meta } : null,
      };
    }
    getState() {
      const last = this.candles[this.candles.length - 1];
      return {
        playing: this.playing, speed: this.speed,
        dia: last ? last.dia : -1, data: last ? last.data : this._date(0),
        preco: last ? last.close : P0,
        loops: this.anchor ? this.anchor.loops : 0, modo: this.mode,
      };
    }
    play() {
      if (this.playing) return;
      this.playing = true;
      this._restartTimer();
      this._emit('state', this.getState());
    }
    pause() {
      if (!this.playing) return;
      this.playing = false;
      clearInterval(this.timer); this.timer = null;
      this._emit('state', this.getState());
    }
    toggle() { this.playing ? this.pause() : this.play(); }
    setSpeed(x) {
      const v = Number(x);
      if (!(v > 0)) return;
      this.speed = v;
      if (this.playing) this._restartTimer();
      this._emit('state', this.getState());
    }
    _restartTimer() {
      clearInterval(this.timer);
      this.timer = setInterval(() => this.advance(1), Math.max(30, 500 / this.speed));
    }
    advance(days) {
      let n = Math.floor(Number(days));
      if (!(n >= 1)) n = 1;
      n = Math.min(n, MAX_ADVANCE);
      const candles = [], ops = [];
      for (let i = 0; i < n; i++) {
        const r = this._step();
        candles.push(r.candle);
        for (const o of r.ops) ops.push(o);
      }
      this._emitUpdate(candles, ops);
    }

    /* ── bots ────────────────────────────────────────────── */
    addBot({ nome, perfil, capital } = {}) {
      const name = String(nome == null ? '' : nome).trim();
      const cap = Number(capital);
      if (!name) throw new Error('Dê um nome ao bot.');
      if (name.length > 20) throw new Error('O nome pode ter no máximo 20 caracteres.');
      if (!PROFILES[perfil]) throw new Error('Escolha um perfil: agressivo, moderado, seguro ou IA.');
      if (perfil === 'ia' && !(global.AgenteIA && global.AgenteIA.pronto())) throw new Error('A rede não foi carregada (falta data/agente_dqn.js).');
      if (!(cap >= 100 && cap <= 10000000)) throw new Error('O capital deve estar entre R$ 100 e R$ 10.000.000.');
      const bot = this._createBot({ nome: name, perfil, capital: cap });
      this._emitUpdate([], []);
      return this._view(bot);
    }
    _createBot({ nome, perfil, capital }) {
      const dia = this.candles.length - 1; // -1 se ainda não houve nenhum dia
      const bot = {
        id: ++this._botId, nome, perfil, capital,
        cash: capital, units: 0, avgCost: 0, opCount: 0,
        acoes: [0, 0, 0], ultima: null, // só o bot de IA usa: contagem de ações escolhidas e a última decisão da rede
        // pequena "personalidade" para bots do mesmo perfil não serem idênticos
        tj: 0.8 + 0.4 * this.botRng(), fj: 0.85 + 0.3 * this.botRng(),
        startDay: Math.max(0, dia), hist: [],
      };
      if (dia >= 0) {
        bot.hist.push({ dia, data: this._date(dia), equity: capital, pos: 0, units: 0, ops: 0, lucro_pct: 0 });
      }
      this.bots.push(bot);
      return bot;
    }
    _view(b) {
      const last = b.hist[b.hist.length - 1];
      const equity = last ? last.equity : b.capital;
      return {
        id: b.id, nome: b.nome, perfil: b.perfil, perfil_label: PROFILES[b.perfil].label,
        capital_inicial: b.capital, saldo: equity, caixa: b.cash, unidades: b.units,
        posicao: last ? last.pos : 0, lucro_pct: (equity / b.capital - 1) * 100,
        operacoes: b.opCount,
      };
    }

    /* ── consultas ───────────────────────────────────────── */
    getCandles() { return this.candles.slice(); }
    getBots() { return this.bots.map((b) => this._view(b)); }
    /** Estado do bot de IA para a tela "Rede neural": última decisão da rede e quantas vezes cada ação foi escolhida. */
    getIAState(botId) {
      const b = this.bots.find((x) => x.id === botId && x.perfil === 'ia');
      return b ? { ultima: b.ultima, acoes: b.acoes.slice() } : null;
    }
    getOperations(limit = 100) { return this.ops.slice(-limit); }
    getOpsCount() { return this._opId; }
    getSeries(botId) {
      const b = this.bots.find((x) => x.id === botId);
      return b ? b.hist : [];
    }
    getTotalSeries(n = 40) {
      const last = this.candles.length - 1;
      const from = Math.max(0, last - n + 1);
      const out = [];
      for (let d = from; d <= last; d++) {
        let equity = 0, capital = 0;
        for (const b of this.bots) {
          const i = d - b.startDay;
          if (i >= 0 && i < b.hist.length) { equity += b.hist[i].equity; capital += b.capital; }
        }
        out.push({ dia: d, equity, capital });
      }
      return out;
    }

    /* ── motor: um dia de mercado ────────────────────────── */
    _nextRegime(cur) {
      const r = this.rng();
      let acc = 0;
      for (const [name, p] of TRANS[cur]) { acc += p; if (r < acc) return name; }
      return cur;
    }

    /** Modo 'regimes': retorno vem direto do regime + leve reversão ao preço inicial. */
    _priceRegimes(prevClose) {
      const R = REGIMES[this.regime];
      const reversion = -0.0015 * Math.log(prevClose / P0);
      const ret = R.mu + reversion + R.sigma * gauss(this.rng);
      this._pendingFlow = 0;
      return { close: r2(prevClose * Math.exp(ret)), ret, sigma: R.sigma, wick: R.sigma * 0.5 };
    }

    /** Modo 'anchor': preço = V (dólar reescalado) × exp(desvio). */
    _priceAnchor(dia, prevClose) {
      const a = this.anchor, reg = this.regime;
      if (dia > 0) {
        let i = a.i + 1, dl = 0;
        if (i >= a.rows.length) { i = 0; a.loops += 1; } // acabou a série: recomeça (sem salto em V)
        else dl = a.ln[i] - a.ln[a.i];
        a.i = i; a.lnV += dl;
      }
      const flow = this._pendingFlow; this._pendingFlow = 0; // fluxo dos bots de ontem
      const impact = this.impact.lambda * flow / this.impact.depth;
      const eps = ANCHOR.sigma[reg] * gauss(this.rng);
      a.d = (1 - this.k) * a.d + ANCHOR.mu[reg] * this.trendMult + eps + impact;
      const close = r2(Math.exp(a.lnV + a.d));
      return {
        close, ret: Math.log(close / prevClose), sigma: ANCHOR.refSigma, wick: 0.004,
        ancora: r2(Math.exp(a.lnV)), ancoraData: a.rows[a.i].data,
      };
    }

    _step() {
      const dia = this.candles.length;
      const prevClose = dia ? this.candles[dia - 1].close : P0;
      if (dia > 0) this.regime = this._nextRegime(this.regime);
      const p = this.mode === 'anchor' ? this._priceAnchor(dia, prevClose) : this._priceRegimes(prevClose);
      const open = r2(prevClose);
      const close = p.close;
      const high = r2(Math.max(open, close) * (1 + Math.abs(gauss(this.rng)) * p.wick));
      const low = r2(Math.min(open, close) * (1 - Math.abs(gauss(this.rng)) * p.wick));
      const volume = Math.round(1000 * (1 + Math.abs(p.ret) / p.sigma) * (0.7 + 0.6 * this.rng()));
      const candle = { dia, data: this._date(dia), open, high, low, close, volume, regime: this.regime };
      if (p.ancora != null) { candle.ancora = p.ancora; candle.ancora_data = p.ancoraData; }
      this.candles.push(candle);

      const newOps = [];
      for (const bot of this.bots) {
        this._act(bot, candle, newOps);
        const equity = bot.cash + bot.units * close;
        bot.hist.push({
          dia, data: candle.data, equity, pos: bot.units * close, units: bot.units,
          ops: bot.opCount, lucro_pct: (equity / bot.capital - 1) * 100,
        });
      }
      for (const o of newOps) this.ops.push(o);
      if (this.ops.length > 20000) this.ops.splice(0, this.ops.length - 20000);
      return { candle, ops: newOps };
    }

    /** Regra do perfil. Ação inválida ou pequena demais vira "manter" (nunca dá erro). */
    _act(bot, c, out) {
      if (bot.perfil === 'ia') return this._actIA(bot, c, out);
      const n = this.candles.length;
      if (n < 20) return; // precisa de histórico para a média de 20 dias
      const base = PROFILES[bot.perfil];
      const P = { ...base, thr: base.thr * bot.tj, fracBuy: Math.min(0.95, base.fracBuy * bot.fj) };
      const sma = (k) => {
        let s = 0;
        for (let i = n - k; i < n; i++) s += this.candles[i].close;
        return s / k;
      };
      const signal = sma(5) / sma(20) - 1;
      const price = c.close;
      const equity = bot.cash + bot.units * price;

      let act = null;
      if (bot.units > 0 && bot.avgCost > 0 && price < bot.avgCost * (1 - P.stop)) {
        act = { tipo: 'venda', frac: 1, motivo: 'stop de perda' };
      } else if (signal > P.thr) {
        act = { tipo: 'compra', frac: P.fracBuy, motivo: 'tendência de alta' };
      } else if (signal < -P.thr) {
        act = { tipo: 'venda', frac: P.fracSell, motivo: 'tendência de baixa' };
      } else if (P.impulse && this.botRng() < P.impulse) {
        act = this.botRng() < 0.5
          ? { tipo: 'compra', frac: 0.25, motivo: 'impulso' }
          : { tipo: 'venda', frac: 0.25, motivo: 'impulso' };
      }
      if (!act) return;

      // slippage: meio spread + impacto proporcional ao tamanho da ordem
      const I = this.impact;
      const slip = (valor) => I.half + (I.lambda * valor) / I.depth;
      let qty, fee, px;
      if (act.tipo === 'compra') {
        const spend = bot.cash * act.frac;
        if (spend < 1 || spend < 0.01 * equity) return;
        px = price * (1 + slip(spend));
        fee = spend * FEE;
        qty = (spend - fee) / px;
        bot.avgCost = (bot.avgCost * bot.units + px * qty) / (bot.units + qty);
        bot.cash -= spend;
        bot.units += qty;
        this._pendingFlow += spend;
      } else {
        qty = bot.units * act.frac;
        const valor = qty * price;
        if (valor < 1 || valor < 0.01 * equity) return;
        px = price * (1 - slip(valor));
        const gross = qty * px;
        fee = gross * FEE;
        bot.cash += gross - fee;
        bot.units -= qty;
        if (bot.units < 1e-9) { bot.units = 0; bot.avgCost = 0; }
        this._pendingFlow -= valor;
      }
      bot.opCount += 1;
      out.push({
        id: ++this._opId, dia: c.dia, data: c.data, botId: bot.id, botNome: bot.nome,
        tipo: act.tipo, quantidade: r4(qty), preco: r4(px), taxa: r2(fee), motivo: act.motivo,
      });
    }

    /**
     * Bot de IA: a rede escolhe uma EXPOSIÇÃO-ALVO e a conta ajusta até lá. Mesmas regras da conta do Python
     * (market/conta.py): taxa + slippage com teto, banda sem operar, alvo 0 zera, sem alavancagem.
     */
    _actIA(bot, c, out) {
      const A = global.AgenteIA;
      const n = this.candles.length;
      if (!A || !A.pronto() || n < A.LOOKBACK + 1) return;
      const price = c.close;
      const equity = bot.cash + bot.units * price;
      const expo = (bot.units * price) / equity;
      const lucro = bot.units > 0 && bot.avgCost > 0 ? price / bot.avgCost - 1 : 0;
      const closes = this.candles.slice(n - A.LOOKBACK - 1).map((x) => x.close);
      const d = A.decidir(closes, expo, lucro);
      const alvo = d.alvo;
      bot.acoes[d.acao] += 1;
      bot.ultima = { dia: c.dia, data: c.data, features: d.features, notas: d.notas, acao: d.acao, alvo, expo, lucro };
      const zerar = alvo === 0 && bot.units > 0;
      if (!zerar && Math.abs(alvo - expo) < IA_BANDA) return;
      const dif = zerar ? -bot.units * price : alvo * equity - bot.units * price;
      const I = this.impact;
      const slip = (valor) => Math.min(I.half + (I.lambda * valor) / I.depth, 0.5);
      const motivo = 'rede neural: alvo ' + Math.round(alvo * 100) + '% (era ' + Math.round(expo * 100) + '%)';
      let tipo, qty, fee, px;
      if (dif > 0) {
        const spend = Math.min(bot.cash, dif);
        if (spend < 1) return;
        px = price * (1 + slip(spend));
        fee = spend * FEE;
        qty = (spend - fee) / px;
        bot.avgCost = (bot.avgCost * bot.units + px * qty) / (bot.units + qty);
        bot.cash -= spend;
        bot.units += qty;
        this._pendingFlow += spend;
        tipo = 'compra';
      } else {
        const valor = zerar ? bot.units * price : Math.min(-dif, bot.units * price);
        if (valor < 1) return;
        qty = zerar ? bot.units : valor / price;
        px = price * (1 - slip(valor));
        const gross = qty * px;
        fee = gross * FEE;
        bot.cash += gross - fee;
        bot.units = zerar || bot.units - qty < 1e-12 ? 0 : bot.units - qty;
        if (bot.units === 0) bot.avgCost = 0;
        this._pendingFlow -= valor;
        tipo = 'venda';
      }
      bot.opCount += 1;
      out.push({
        id: ++this._opId, dia: c.dia, data: c.data, botId: bot.id, botNome: bot.nome,
        tipo, quantidade: r4(qty), preco: r4(px), taxa: r2(fee), motivo,
      });
    }
  }

  global.UdiProvider = {
    PROFILES,
    create: (opts) => new MockProvider(opts),
  };
})(typeof window !== 'undefined' ? window : globalThis);
