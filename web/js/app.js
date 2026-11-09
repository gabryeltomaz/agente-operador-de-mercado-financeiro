/**
 * udiCOIN Market — interface.
 * Só usa a interface do provedor (UdiProvider). Para ligar o motor Python,
 * troque `UdiProvider.create(...)` por um provedor que fale com a API.
 */
(function () {
  'use strict';

  if (!window.UdiProvider || !window.LightweightCharts || !window.Chart) {
    console.error('Faltam bibliotecas (UdiProvider, LightweightCharts ou Chart.js). Verifique a conexão com a internet.');
    return;
  }

  /* ── utilidades ─────────────────────────────────────────────── */
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const nfBRL = new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL' });
  const nfBRL0 = new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL', maximumFractionDigits: 0 });
  const nf2 = new Intl.NumberFormat('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const nf4 = new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 4 });
  const nf0 = new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 0 });
  const fmtBRL = (n) => nfBRL.format(n);
  const fmtPct = (n) => (n >= 0 ? '+' : '−') + nf2.format(Math.abs(n)) + '%';
  const fmtArrowPct = (n) => (n >= 0 ? '▲ ' : '▼ ') + nf2.format(Math.abs(n)) + '%';
  const tone = (n) => (n >= 0 ? 'text-up' : 'text-down');
  const brDate = (iso) => iso.slice(8, 10) + '/' + iso.slice(5, 7) + '/' + iso.slice(0, 4);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function setActive(container, attr, value) {
    $$('button', container).forEach((b) => {
      const on = b.dataset[attr] === String(value);
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
  }

  function sparkline(values, color) {
    if (values.length < 2) return '';
    const w = 96, h = 34, min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
    const pts = values.map((v, i) => `${((i / (values.length - 1)) * w).toFixed(1)},${(h - 3 - ((v - min) / span) * (h - 6)).toFixed(1)}`).join(' ');
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
  }

  /* ── dados ──────────────────────────────────────────────────── */
  // Âncora no dólar real, se o arquivo data/usdbrl.js foi gerado (tools/baixar_dolar.py).
  const anchorRows = Array.isArray(window.USDBRL) && window.USDBRL.length > 300 ? window.USDBRL : null;
  // IA (DQN): se data/agente_dqn.js existe, o mercado usa a mesma força de tendência em que a rede foi treinada.
  // Dá para forçar outra com  ?tendencia=1  (1 = realista) na URL.
  const IA = window.AgenteIA && window.AgenteIA.pronto() ? window.AgenteIA : null;
  const qsTend = Number(new URLSearchParams(location.search).get('tendencia'));
  const trendMult = qsTend > 0 ? qsTend : (IA ? IA.nivel() || 1 : 1);
  let sim, anchorProblem = '';
  try {
    sim = window.UdiProvider.create({
      seed: 7, mode: anchorRows ? 'anchor' : 'regimes', anchorData: anchorRows, anchorMeta: window.USDBRL_META,
      trendMult,
    });
  } catch (err) {
    console.warn('Âncora indisponível, usando o modo simulado:', err.message);
    anchorProblem = err.message;
    sim = window.UdiProvider.create({ seed: 7, mode: 'regimes' });
  }
  const PROFILES = window.UdiProvider.PROFILES;
  const CFG = sim.getConfig();
  const ANCHORED = CFG.mode === 'anchor';
  const SPY = CFG.stepsPerYear; // passos por ano (252 dias úteis com âncora, 365 sem)

  $('#adv-year').dataset.adv = String(SPY);
  $('#adv-unit option:last-child').value = String(SPY);
  if (ANCHORED) {
    $('#side-source').textContent = 'dólar real (BCB) + simulação';
    $('#btn-anchor').classList.remove('hidden');
  }
  function renderPill() {
    const pill = $('#mode-pill');
    if (ANCHORED) {
      const a = CFG.anchor, loops = sim.getState().loops;
      pill.textContent = `Âncora: dólar ${a.from.slice(0, 4)}–${a.to.slice(0, 4)}` + (loops ? ` · série reiniciada ${loops}×` : '') + (CFG.trendMult !== 1 ? ` · tendências ${CFG.trendMult}×` : '');
      pill.title = 'O preço segue o dólar real (reescalado) mais um desvio fictício. Um passo = um dia útil.';
      pill.className = 'ml-2 px-2 py-0.5 rounded-full border border-up/40 text-[10px] text-up';
    } else {
      pill.textContent = 'Modo simulado (sem âncora)';
      pill.title = anchorProblem || 'Rode tools/baixar_dolar.py para ancorar a udiCOIN no dólar real.';
      pill.className = 'ml-2 px-2 py-0.5 rounded-full border border-line text-[10px] text-mute';
    }
  }

  const BOT_COLORS = ['#38BDF8', '#A78BFA', '#FBBF24', '#F472B6', '#2DD4BF', '#FB923C', '#94A3B8', '#C084FC'];
  const colors = new Map();
  const colorOf = (id) => {
    if (!colors.has(id)) colors.set(id, BOT_COLORS[colors.size % BOT_COLORS.length]);
    return colors.get(id);
  };

  let candleList = sim.getCandles();
  const candleByDate = new Map(candleList.map((c) => [c.data, c]));
  let selected = new Set(sim.getBots().map((b) => { colorOf(b.id); return b.id; }));
  let view = 'equity';
  let tab = 'log';
  let lastOpIdRendered = sim.getOpsCount();

  /* ── gráfico de preço (Lightweight Charts) ─────────────────── */
  const UP = '#00E676', DOWN = '#FF4545';
  const chart = LightweightCharts.createChart($('#chart'), {
    autoSize: true,
    layout: { background: { type: 'solid', color: '#0B0B0B' }, textColor: '#8491A3', fontFamily: 'JetBrains Mono, monospace', fontSize: 11 },
    grid: { vertLines: { color: '#11171c' }, horzLines: { color: '#11171c' } },
    rightPriceScale: { borderColor: '#1F2933' },
    timeScale: { borderColor: '#1F2933', rightOffset: 4 },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });
  const candles = chart.addCandlestickSeries({
    upColor: UP, downColor: DOWN, borderUpColor: UP, borderDownColor: DOWN, wickUpColor: UP, wickDownColor: DOWN,
  });
  const line = chart.addLineSeries({ color: UP, lineWidth: 2, visible: false });
  const anchorLine = chart.addLineSeries({
    color: '#8491A3', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, visible: false,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
  });
  const vol = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol', lastValueVisible: false, priceLineVisible: false });
  chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });

  const toCandle = (c) => ({ time: c.data, open: c.open, high: c.high, low: c.low, close: c.close });
  const toVol = (c) => ({ time: c.data, value: c.volume, color: c.close >= c.open ? 'rgba(0,230,118,.3)' : 'rgba(255,69,69,.3)' });

  candles.setData(candleList.map(toCandle));
  line.setData(candleList.map((c) => ({ time: c.data, value: c.close })));
  vol.setData(candleList.map(toVol));
  if (ANCHORED) anchorLine.setData(candleList.map((c) => ({ time: c.data, value: c.ancora })));
  let anchorOn = false;
  $('#btn-anchor').addEventListener('click', () => {
    anchorOn = !anchorOn;
    anchorLine.applyOptions({ visible: anchorOn });
    $('#btn-anchor').classList.toggle('active', anchorOn);
    $('#btn-anchor').setAttribute('aria-pressed', String(anchorOn));
  });

  function setRange(key) {
    const n = { '1M': 30, '3M': 90, '6M': 180, '1A': 365 }[key] || 0;
    const total = candleList.length;
    if (!n || n >= total) chart.timeScale().fitContent();
    else chart.timeScale().setVisibleLogicalRange({ from: total - n, to: total + 3 });
  }
  setRange('6M');

  function showOHLC(c) {
    if (!c) return;
    const t = tone(c.close - c.open);
    $('#ohlc').innerHTML =
      `<span>Abertura <b class="${t} font-medium">${nf2.format(c.open)}</b></span>` +
      `<span>Máxima <b class="${t} font-medium">${nf2.format(c.high)}</b></span>` +
      `<span>Mínima <b class="${t} font-medium">${nf2.format(c.low)}</b></span>` +
      `<span>Fechamento <b class="${t} font-medium">${nf2.format(c.close)}</b></span>` +
      `<span>${brDate(c.data)}</span>`;
  }
  chart.subscribeCrosshairMove((p) => {
    const c = p && p.time ? candleByDate.get(p.time) : null;
    showOHLC(c || candleList[candleList.length - 1]);
  });

  $('#range-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    setActive($('#range-seg'), 'range', b.dataset.range);
    setRange(b.dataset.range);
  });
  $('#type-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    setActive($('#type-seg'), 'type', b.dataset.type);
    candles.applyOptions({ visible: b.dataset.type === 'candles' });
    line.applyOptions({ visible: b.dataset.type === 'line' });
  });

  /* ── gráficos de estatística (Chart.js) ─────────────────────── */
  Chart.defaults.font.family = "'JetBrains Mono', monospace";
  Chart.defaults.font.size = 11;
  Chart.defaults.color = '#8491A3';

  const VIEWS = {
    equity: { key: 'equity', tick: (v) => nfBRL0.format(v), tip: (v) => fmtBRL(v) },
    pos: { key: 'pos', tick: (v) => nfBRL0.format(v), tip: (v) => fmtBRL(v) + ' em UDI' },
    ops: { key: 'ops', tick: (v) => nf0.format(v), tip: (v) => nf0.format(v) + ' ops' },
  };

  function makeChart(canvas, tick, tip) {
    return new Chart(canvas, {
      type: 'line',
      data: { labels: [], datasets: [] },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false,
        onResize: (c, size) => { c.options.scales.x.ticks.maxTicksLimit = size.width < 520 ? 3 : 7; },
        interaction: { mode: 'index', intersect: false },
        elements: { point: { radius: 0, hoverRadius: 3 }, line: { borderWidth: 2 } },
        plugins: {
          legend: { labels: { color: '#8491A3', boxWidth: 8, boxHeight: 8, usePointStyle: true } },
          tooltip: {
            backgroundColor: '#0B0B0B', borderColor: '#1F2933', borderWidth: 1, titleColor: '#E5E7EB', bodyColor: '#E5E7EB',
            callbacks: { label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y == null ? '—' : tip(ctx.parsed.y)}` },
          },
        },
        scales: {
          x: {
            grid: { color: '#11171c' }, border: { color: '#1F2933' },
            ticks: {
              maxTicksLimit: window.innerWidth < 640 ? 3 : 7, maxRotation: 0,
              callback(v) { const s = this.getLabelForValue(v) || ''; return s.slice(8, 10) + '/' + s.slice(5, 7) + '/' + s.slice(2, 4); },
            },
          },
          y: { grid: { color: '#11171c' }, border: { color: '#1F2933' }, ticks: { callback: (v) => tick(v) } },
        },
      },
    });
  }
  const profitChart = makeChart($('#chart-profit'), (v) => nf0.format(v) + '%', (v) => fmtPct(v));
  const viewChart = makeChart($('#chart-view'), VIEWS.equity.tick, VIEWS.equity.tip);

  function aligned(botId, key, total) {
    const arr = new Array(total).fill(null);
    for (const p of sim.getSeries(botId)) if (p.dia < total) arr[p.dia] = p[key];
    return arr;
  }

  /* ── renderização ───────────────────────────────────────────── */
  function renderHeader() {
    const last = candleList[candleList.length - 1];
    const prev = candleList[candleList.length - 2];
    const chg = prev ? (last.close / prev.close - 1) * 100 : 0;
    $('#price').textContent = fmtBRL(last.close);
    const pv = $('#price-var');
    pv.textContent = fmtArrowPct(chg);
    pv.className = 'num text-sm ' + tone(chg);
    showOHLC(last);

    const st = sim.getState();
    const year = Math.floor(st.dia / SPY) + 1;
    renderPill();
    $('#sim-date').textContent = `${ANCHORED ? 'Dia útil' : 'Dia'} ${st.dia + 1} · Ano ${year} · ${brDate(st.data)}`;
    $('#side-date').textContent = `Dia ${st.dia + 1} · ${brDate(st.data)}`;
  }

  function renderKPIs(bots) {
    const last = candleList[candleList.length - 1];
    const prev = candleList[candleList.length - 2];
    const chg = prev ? (last.close / prev.close - 1) * 100 : 0;
    $('#kpi-price').textContent = fmtBRL(last.close);
    $('#kpi-price-var').innerHTML = `<span class="${tone(chg)}">${fmtArrowPct(chg)}</span> <span class="text-mute">no dia</span>`;
    $('#kpi-price-spark').innerHTML = sparkline(candleList.slice(-40).map((c) => c.close), chg >= 0 ? UP : DOWN);

    const tot = sim.getTotalSeries(40);
    const cur = tot[tot.length - 1];
    const totPct = cur && cur.capital ? (cur.equity / cur.capital - 1) * 100 : 0;
    $('#kpi-equity').textContent = fmtBRL(cur ? cur.equity : 0);
    $('#kpi-equity-var').innerHTML = `<span class="${tone(totPct)}">${fmtArrowPct(totPct)}</span> <span class="text-mute">acumulado</span>`;
    $('#kpi-equity-spark').innerHTML = sparkline(tot.map((t) => t.equity), totPct >= 0 ? UP : DOWN);

    const best = bots.slice().sort((a, b) => b.lucro_pct - a.lucro_pct)[0];
    $('#kpi-best').textContent = best ? best.nome : '—';
    $('#kpi-best-var').innerHTML = best ? `<span class="${tone(best.lucro_pct)}">${fmtArrowPct(best.lucro_pct)}</span> <span class="text-mute">${esc(best.perfil_label)}</span>` : '';

    const total = sim.getOpsCount();
    const today = sim.getOperations(500).filter((o) => o.dia === last.dia).length;
    $('#kpi-ops').textContent = nf0.format(total);
    $('#kpi-ops-sub').textContent = `${today} hoje`;
  }

  function renderLog() {
    const ops = sim.getOperations(60).slice().reverse();
    const prevId = lastOpIdRendered;
    $('#log-list').innerHTML = ops.length ? ops.map((o) => {
      const buy = o.tipo === 'compra';
      return `<li class="log-item ${o.id > prevId ? 'fresh' : ''}">
        <span class="tag ${buy ? 'buy' : 'sell'}">${buy ? 'COMPRA' : 'VENDA'}</span>
        <div class="min-w-0 text-sm"><span class="font-medium">${esc(o.botNome)}</span>
          <span class="text-mute">${buy ? 'comprou' : 'vendeu'}</span>
          <span class="num">${nf2.format(o.quantidade)} UDI</span>
          <span class="text-mute">a</span> <span class="num">${fmtBRL(o.preco)}</span>
          <div class="text-[11px] text-mute">${esc(o.motivo)}</div></div>
        <span class="num text-[11px] text-mute whitespace-nowrap">Dia ${o.dia + 1}</span></li>`;
    }).join('') : '<li class="p-6 text-sm text-mute text-center">Nenhuma operação ainda. Aperte Play ou avance o tempo.</li>';
    lastOpIdRendered = sim.getOpsCount();
    $('#count-log').textContent = nf0.format(sim.getOpsCount());
  }

  function renderTraders(bots) {
    $('#count-traders').textContent = bots.length;
    $('#traders-list').innerHTML = bots.map((b) => {
      const c = colorOf(b.id);
      return `<li class="trader">
        <div class="avatar" style="background:${c}22;color:${c};border-color:${c}66">${esc(b.nome.slice(0, 2).toUpperCase())}</div>
        <div class="min-w-0 flex-1">
          <div class="flex items-center gap-2"><span class="font-medium truncate">${esc(b.nome)}</span><span class="badge badge-${b.perfil}">${esc(b.perfil_label)}</span></div>
          <div class="num text-xs text-mute">${nf4.format(b.unidades)} UDI · ${b.operacoes} ops</div>
        </div>
        <div class="text-right"><div class="num text-sm">${fmtBRL(b.saldo)}</div>
          <div class="num text-xs ${tone(b.lucro_pct)}">${fmtArrowPct(b.lucro_pct)}</div></div></li>`;
    }).join('');
  }

  function renderChips(bots) {
    const all = bots.length > 0 && bots.every((b) => selected.has(b.id));
    $('#chips').innerHTML =
      `<button class="chip ${all ? 'on' : ''}" data-id="all" aria-pressed="${all}">Todos</button>` +
      bots.map((b) => `<button class="chip ${selected.has(b.id) ? 'on' : ''}" data-id="${b.id}" aria-pressed="${selected.has(b.id)}"><i style="background:${colorOf(b.id)}"></i>${esc(b.nome)}</button>`).join('');
  }

  function renderStats(bots) {
    const total = candleList.length;
    const labels = candleList.map((c) => c.data);
    const chosen = bots.filter((b) => selected.has(b.id));
    const dataset = (b, key, fill) => ({
      label: b.nome, data: aligned(b.id, key, total), borderColor: colorOf(b.id),
      backgroundColor: colorOf(b.id) + '22', fill, spanGaps: false,
    });

    profitChart.data.labels = labels;
    profitChart.data.datasets = chosen.map((b) => dataset(b, 'lucro_pct', false));
    profitChart.update('none');

    const V = VIEWS[view];
    viewChart.options.scales.y.ticks.callback = (v) => V.tick(v);
    viewChart.options.plugins.tooltip.callbacks.label = (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y == null ? '—' : V.tip(ctx.parsed.y)}`;
    viewChart.data.labels = labels;
    viewChart.data.datasets = chosen.map((b) => dataset(b, V.key, view === 'equity' && chosen.length === 1));
    viewChart.update('none');

    const sum = chosen.reduce((a, b) => a + b.saldo, 0);
    const cap = chosen.reduce((a, b) => a + b.capital_inicial, 0);
    const p = cap ? (sum / cap - 1) * 100 : 0;
    $('#montante-value').textContent = fmtBRL(sum);
    const mp = $('#montante-pct');
    mp.textContent = fmtArrowPct(p);
    mp.className = 'num text-base mt-1 ' + tone(p);
    $('#montante-names').textContent = chosen.length === bots.length ? 'Todos os bots' : chosen.map((b) => b.nome).join(', ');

    $('#stats-body').innerHTML = bots.map((b) => `
      <tr data-id="${b.id}" class="${selected.has(b.id) ? 'sel' : ''}">
        <td><span class="inline-block w-2 h-2 rounded-full mr-2" style="background:${colorOf(b.id)}"></span>${esc(b.nome)}</td>
        <td><span class="badge badge-${b.perfil}">${esc(b.perfil_label)}</span></td>
        <td class="num text-right">${fmtBRL(b.saldo)}</td>
        <td class="num text-right ${tone(b.lucro_pct)}">${fmtArrowPct(b.lucro_pct)}</td>
        <td class="num text-right">${nf0.format(b.operacoes)}</td>
        <td class="num text-right">${nf4.format(b.unidades)}</td></tr>`).join('');
  }

  let lastStats = 0, statsTimer = 0;
  function renderStatsThrottled() {
    const now = performance.now();
    if (now - lastStats > 250) { lastStats = now; renderStats(sim.getBots()); }
    else if (!statsTimer) {
      statsTimer = setTimeout(() => { statsTimer = 0; lastStats = performance.now(); renderStats(sim.getBots()); }, 250);
    }
  }

  let raf = 0;
  function scheduleRender() {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      const bots = sim.getBots();
      renderHeader();
      renderKPIs(bots);
      renderLog();
      renderTraders(bots);
      renderChips(bots);
      renderStatsThrottled();
    });
  }

  /* ── eventos do provedor ────────────────────────────────────── */
  sim.on('update', (p) => {
    for (const c of p.candles) {
      candleList.push(c);
      candleByDate.set(c.data, c);
      candles.update(toCandle(c));
      line.update({ time: c.data, value: c.close });
      vol.update(toVol(c));
      if (ANCHORED) anchorLine.update({ time: c.data, value: c.ancora });
    }
    p.bots.forEach((b) => colorOf(b.id));
    scheduleRender();
  });

  function renderState(s) {
    $('#btn-play').classList.toggle('active', s.playing);
    $('#btn-pause').classList.toggle('active-down', !s.playing);
    $('#btn-play').setAttribute('aria-pressed', String(s.playing));
    $('#btn-pause').setAttribute('aria-pressed', String(!s.playing));
    setActive($('#speed-seg'), 'speed', s.speed);
    $('#sim-status').textContent = s.playing ? 'Simulação em andamento' : 'Pausada';
    $('#sim-status').className = s.playing ? 'text-up' : 'text-mute';
    $('#sim-dot').className = 'inline-block w-2 h-2 rounded-full ' + (s.playing ? 'bg-up live' : 'bg-mute');
  }
  sim.on('state', renderState);

  /* ── controles ──────────────────────────────────────────────── */
  $('#btn-play').addEventListener('click', () => sim.play());
  $('#btn-pause').addEventListener('click', () => sim.pause());
  $('#speed-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (b) sim.setSpeed(b.dataset.speed);
  });
  $('#adv-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-adv]'); if (b) sim.advance(Number(b.dataset.adv));
  });
  $('#adv-go').addEventListener('click', () => {
    const n = Math.floor(Number($('#adv-n').value));
    if (!(n >= 1)) { $('#adv-n').focus(); return; }
    sim.advance(n * Number($('#adv-unit').value));
  });
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); sim.toggle(); }
  });

  /* ── abas ───────────────────────────────────────────────────── */
  function setTab(t) {
    tab = t;
    $$('.tabbtn').forEach((b) => {
      const on = b.dataset.tab === t;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', String(on));
    });
    $('#pane-log').classList.toggle('hidden', t !== 'log');
    $('#pane-traders').classList.toggle('hidden', t !== 'traders');
  }
  $$('.tabbtn').forEach((b) => b.addEventListener('click', () => setTab(b.dataset.tab)));

  /* ── adicionar bot ──────────────────────────────────────────── */
  if (IA) { const bIA = $('#bot-profile-ia'); if (bIA) bIA.style.display = ''; }
  let formProfile = 'moderado';
  function describeProfile() { $('#bot-profile-desc').textContent = PROFILES[formProfile].desc; }
  describeProfile();

  $('#bot-profile-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    formProfile = b.dataset.profile;
    setActive($('#bot-profile-seg'), 'profile', formProfile);
    describeProfile();
  });
  function openForm(open) {
    $('#bot-form').classList.toggle('hidden', !open);
    $('#bot-add-btn').classList.toggle('hidden', open);
    $('#bot-error').classList.add('hidden');
    if (open) { $('#bot-name').value = ''; $('#bot-name').focus(); }
  }
  $('#bot-name').addEventListener('input', () => $('#bot-error').classList.add('hidden'));
  $('#bot-add-btn').addEventListener('click', () => openForm(true));
  $('#bot-cancel').addEventListener('click', () => openForm(false));
  $('#bot-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const wasAll = sim.getBots().every((b) => selected.has(b.id));
    try {
      const bot = sim.addBot({ nome: $('#bot-name').value, perfil: formProfile, capital: $('#bot-capital').value });
      colorOf(bot.id);
      if (wasAll) selected.add(bot.id);
      openForm(false);
    } catch (err) {
      const box = $('#bot-error');
      box.textContent = err.message;
      box.classList.remove('hidden');
    }
  });

  /* ── seleção de bots e visualização ─────────────────────────── */
  $('#chips').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    const bots = sim.getBots();
    if (b.dataset.id === 'all') selected = new Set(bots.map((x) => x.id));
    else {
      const id = Number(b.dataset.id);
      const allOn = bots.every((x) => selected.has(x.id));
      if (allOn) selected = new Set([id]);
      else if (selected.has(id)) { if (selected.size > 1) selected.delete(id); }
      else selected.add(id);
    }
    renderChips(bots); renderStats(bots);
  });
  $('#stats-body').addEventListener('click', (e) => {
    const tr = e.target.closest('tr'); if (!tr) return;
    selected = new Set([Number(tr.dataset.id)]);
    const bots = sim.getBots(); renderChips(bots); renderStats(bots);
  });
  $('#view-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    view = b.dataset.view;
    setActive($('#view-seg'), 'view', view);
    renderStats(sim.getBots());
  });

  /* ── início ─────────────────────────────────────────────────── */
  renderState(sim.getState());
  setActive($('#range-seg'), 'range', '6M');
  setActive($('#type-seg'), 'type', 'candles');
  setActive($('#view-seg'), 'view', 'equity');
  setActive($('#bot-profile-seg'), 'profile', formProfile);
  const bots0 = sim.getBots();
  renderHeader(); renderKPIs(bots0); renderLog(); renderTraders(bots0); renderChips(bots0); renderStats(bots0);
  setTab('log');
  if (window.RedeView) window.RedeView.init({ sim, IA: window.AgenteIA && window.AgenteIA.pronto() ? window.AgenteIA : null, setActive });

  window.udicoin = { sim }; // atalho para testar no console: udicoin.sim.advance(30)
})();
