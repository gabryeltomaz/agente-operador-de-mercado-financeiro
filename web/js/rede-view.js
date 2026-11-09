/**
 * Tela "Rede neural": mostra a rede do bot de IA (DQN) sendo ativada ao vivo, a cada dia simulado.
 *
 * O que é "ao vivo" aqui: a cada dia a rede recebe 7 números e calcula 3 notas. A tela mostra o caminho
 * da informação: qual entrada pesa, quais neurônios acendem e qual nota vence. Os PESOS estão congelados
 * (o aprendizado aconteceu no Python); o que se move é a ativação. O aprendizado aparece na curva de treino.
 *
 * Usa só: provedor (sim.getBots, sim.getIAState, sim.on), AgenteIA (ativacoes/info/pesos) e Chart.js (curva).
 */
(function (global) {
  'use strict';

  const $ = (s, r = document) => r.querySelector(s);
  const nf = (d) => new Intl.NumberFormat('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
  const f2 = nf(2), f3 = nf(3), f0 = nf(0);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

  const NOMES = ['retorno 1d', 'retorno 5d', 'retorno 20d', 'médias 5/20', 'volatilidade', 'exposição', 'lucro posição'];
  const DICAS = [
    'Quanto o preço variou no último dia (em desvios-padrão).',
    'Variação dos últimos 5 dias.',
    'Variação dos últimos 20 dias (a tendência mais longa que a rede vê).',
    'Média dos 5 últimos dias contra a média de 20: positivo = tendência de alta recente.',
    'O quanto o preço andou agitado nas últimas 20 sessões.',
    'Quanto do patrimônio da própria rede está em udiCOIN agora (0 a 100%).',
    'Ganho ou perda não realizada da posição aberta.',
  ];
  const VERDE = [0, 230, 118], VERM = [255, 69, 69], AZUL = [100, 181, 246], LARANJA = [255, 138, 101], CINZA = [27, 36, 43];
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

  const S = {
    sim: null, IA: null, setActive: null,
    inited: false, visible: false, raf: 0,
    botId: null, target: null, disp: null, ultima: null, acoes: [0, 0, 0],
    dirty: true, chart: null, heatDone: false, estatico: false,
  };

  /* ── montagem da tela ───────────────────────────────────────── */
  function montarHTML() {
    $('#view-rede').innerHTML = `
      <div class="panel p-4">
        <div class="flex flex-wrap items-center gap-x-4 gap-y-2">
          <div>
            <h2 class="text-base font-semibold">Rede neural <span class="text-mute font-normal">· DQN</span></h2>
            <p class="text-xs text-mute mt-0.5">O caminho da informação, dia a dia: 7 entradas → 2 camadas ocultas → 3 notas (uma por ação).</p>
          </div>
          <div class="ml-auto flex flex-wrap items-center gap-2">
            <select id="rn-bot" class="field !w-40 !h-8" aria-label="Bot de IA"></select>
            <button id="rn-play" class="btn" type="button">Play</button>
            <button id="rn-step" class="btn" type="button">+1 dia</button>
            <div class="seg" id="rn-speed" role="group" aria-label="Velocidade">
              <button data-speed="1" class="active">1x</button><button data-speed="2">2x</button>
              <button data-speed="5">5x</button><button data-speed="10">10x</button>
            </div>
            <span id="rn-date" class="num text-xs text-mute"></span>
          </div>
        </div>
        <div id="rn-aviso" class="hidden mt-3 text-sm text-down"></div>
        <canvas id="rn-canvas" class="block w-full mt-3" style="height:470px" aria-label="Diagrama da rede neural em atividade"></canvas>
        <div class="flex flex-wrap gap-x-5 gap-y-1 mt-2 text-[11px] text-mute">
          <span><span class="inline-block w-2 h-2 rounded-full align-middle" style="background:#00e676"></span> neurônio ativo (quanto mais brilho, mais forte)</span>
          <span><span class="inline-block w-2 h-2 rounded-full align-middle" style="background:#1b242b;border:1px solid #2b3844"></span> neurônio desligado (ReLU = 0)</span>
          <span><span class="inline-block w-4 h-0.5 align-middle" style="background:#00e676"></span> conexão que empurra para cima</span>
          <span><span class="inline-block w-4 h-0.5 align-middle" style="background:#ff4545"></span> conexão que empurra para baixo</span>
          <span><span class="inline-block w-2 h-2 rounded-full align-middle" style="background:#64b5f6"></span>/<span class="inline-block w-2 h-2 rounded-full align-middle" style="background:#ff8a65"></span> entrada negativa / positiva</span>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div class="panel p-4 min-w-0">
          <h3 class="text-sm font-semibold">Decisão de hoje</h3>
          <p id="rn-decisao" class="text-sm mt-1 text-mute">—</p>
          <div class="text-xs text-mute mt-4 mb-1 uppercase tracking-wider">O que a rede viu (entradas)</div>
          <div id="rn-feat" class="space-y-1.5"></div>
          <div class="text-xs text-mute mt-4 mb-1 uppercase tracking-wider">Notas das ações (valor Q)</div>
          <div id="rn-notas" class="space-y-1.5"></div>
          <p class="text-[11px] text-mute mt-3">A nota Q estima o retorno futuro (já descontado) de cada ação. A rede escolhe a maior.</p>
        </div>

        <div class="panel p-4 min-w-0">
          <h3 class="text-sm font-semibold">Arquitetura</h3>
          <div id="rn-arq" class="mt-2 text-sm space-y-1"></div>
          <div class="text-xs text-mute mt-4 mb-1 uppercase tracking-wider">Neurônios ativos hoje</div>
          <div id="rn-ativos" class="space-y-1.5"></div>
          <div class="text-xs text-mute mt-4 mb-1 uppercase tracking-wider">Ações escolhidas até agora</div>
          <div id="rn-dist" class="space-y-1.5"></div>
        </div>

        <div class="panel p-4 min-w-0">
          <h3 class="text-sm font-semibold">Como foi treinada</h3>
          <dl id="rn-treino" class="mt-2 text-sm"></dl>
        </div>
      </div>

      <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div class="panel p-4 min-w-0">
          <h3 class="text-sm font-semibold">Curva de aprendizado <span class="text-mute font-normal">· treino no Python</span></h3>
          <p class="text-xs text-mute mt-1">Retorno da rede em episódios de validação (que ela nunca viu) ao longo do treino, contra "comprar e segurar".</p>
          <div class="h-64 mt-2"><canvas id="rn-curva"></canvas></div>
          <p id="rn-curva-msg" class="hidden text-sm text-mute mt-2"></p>
        </div>
        <div class="panel p-4 min-w-0">
          <h3 class="text-sm font-semibold">Mapa dos pesos da 1ª camada</h3>
          <p class="text-xs text-mute mt-1">Cada coluna é um neurônio; cada linha, uma entrada. <span class="text-up">Verde</span>: a entrada acende o neurônio. <span class="text-down">Vermelho</span>: a entrada o desliga.</p>
          <canvas id="rn-heat" class="block w-full mt-3" style="height:170px"></canvas>
        </div>
      </div>

      <div class="panel p-4 text-sm text-mute">
        <span class="text-ink font-medium">Como ler esta tela.</span>
        Aqui a rede está <b>em uso</b>: os pesos estão fixos desde o treino (feito no Python, em simulação). O que se move ao vivo é a ativação dos
        neurônios, que muda a cada dia conforme o preço e a conta do bot. O aprendizado em si aparece na curva acima. Rede pequena (milhares de pesos),
        treinada num mercado com tendências reforçadas: ela mostra a ideia do DQN, não uma previsão do mercado real.
      </div>`;
  }

  /* ── dados ──────────────────────────────────────────────────── */
  function botsIA() { return S.sim.getBots().filter((b) => b.perfil === 'ia'); }

  function popularSelect() {
    const sel = $('#rn-bot');
    const bots = botsIA();
    const atual = sel.value;
    sel.innerHTML = bots.map((b) => `<option value="${b.id}">${esc(b.nome)}</option>`).join('');
    if (bots.length) {
      const ok = bots.some((b) => String(b.id) === atual);
      sel.value = ok ? atual : String(bots[0].id);
      S.botId = Number(sel.value);
    } else S.botId = null;
  }

  function lerEstado() {
    if (S.botId == null || !S.sim.getIAState) return;
    const st = S.sim.getIAState(S.botId);
    if (!st) return;
    S.acoes = st.acoes;
    if (!st.ultima) { S.ultima = null; S.target = null; return; }
    if (S.ultima && S.ultima.dia === st.ultima.dia && S.target) return; // nada novo
    S.ultima = st.ultima;
    S.target = S.IA.ativacoes(st.ultima.features);
    if (!S.disp) S.disp = S.target.map((l) => l.slice());
    S.dirty = true;
  }

  function onUpdate() {
    if (!S.inited || !S.IA) return;
    if (!S.visible) return; // fora da tela: nada a fazer
    popularSelectSeNecessario();
    lerEstado();
  }
  function popularSelectSeNecessario() {
    if (botsIA().length !== $('#rn-bot').options.length) popularSelect();
  }

  /* ── desenho do diagrama ────────────────────────────────────── */
  function layout(W, H) {
    const narrow = W < 640;
    const padL = narrow ? 88 : 112, padR = narrow ? 70 : 168, top = 34, bot = 22;
    const usable = W - padL - padR;
    const xs = [0, 1, 2, 3].map((i) => padL + (usable * i) / 3);
    const tam = S.IA.info().tamanhos;
    const ys = tam.map((n, l) => {
      const arr = [];
      if (l === 0 || l === tam.length - 1) {
        const gap = Math.min(l === 0 ? 52 : 110, (H - top - bot) / Math.max(1, n - 1));
        const total = gap * (n - 1), y0 = top + (H - top - bot - total) / 2;
        for (let i = 0; i < n; i++) arr.push(y0 + gap * i);
      } else {
        const gap = (H - top - bot) / (n - 1);
        for (let i = 0; i < n; i++) arr.push(top + gap * i);
      }
      return arr;
    });
    return { xs, ys, tam, narrow };
  }

  function desenhar() {
    const cv = $('#rn-canvas');
    const dpr = global.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const { xs, ys, tam, narrow } = layout(W, H);
    const camadas = S.IA.pesos().camadas;
    const nomesAcao = S.IA.pesos().niveis.map((n) => (n === 0 ? 'Fora' : n === 1 ? 'Tudo' : 'Meio'));

    ctx.font = '600 11px Inter, system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    const titulos = narrow ? ['Entrada', 'Oculta 1', 'Oculta 2', 'Saída']
      : ['Entrada (' + tam[0] + ')', 'Oculta 1 (' + tam[1] + ')', 'Oculta 2 (' + tam[2] + ')', 'Saída (' + tam[3] + ' notas)'];
    if (narrow) ctx.font = '600 10px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#8491a3'; ctx.textAlign = 'center';
    titulos.forEach((t, l) => ctx.fillText(t, xs[l], 14));

    const D = S.disp;
    if (!D) {
      ctx.fillStyle = '#8491a3'; ctx.font = '13px Inter, system-ui, sans-serif';
      ctx.fillText('Aguardando a primeira decisão da rede (precisa de 21 dias de histórico)…', W / 2, H / 2);
    }

    // conexões: só as que carregam mais sinal (|peso × ativação| alto), para o desenho ficar legível
    if (D) {
      for (let l = 0; l < camadas.length; l++) {
        const Wm = camadas[l].W, a = D[l];
        const lista = [];
        let maxAbs = 1e-9;
        for (let j = 0; j < Wm.length; j++) {
          const w = Wm[j];
          for (let k = 0; k < w.length; k++) {
            const c = w[k] * a[k];
            const ab = Math.abs(c);
            if (ab > maxAbs) maxAbs = ab;
            lista.push([k, j, c, ab]);
          }
        }
        const lim = l === 1 ? 0.38 : 0.22, cap = l === 1 ? 420 : 600;
        let sel = lista.filter((e) => e[3] >= lim * maxAbs);
        if (sel.length > cap) { sel.sort((p, q) => q[3] - p[3]); sel = sel.slice(0, cap); }
        ctx.lineCap = 'round';
        for (const [k, j, c, ab] of sel) {
          const rel = ab / maxAbs;
          ctx.strokeStyle = rgba(c >= 0 ? VERDE : VERM, 0.08 + 0.6 * rel);
          ctx.lineWidth = 0.4 + 1.5 * rel;
          ctx.beginPath(); ctx.moveTo(xs[l], ys[l][k]); ctx.lineTo(xs[l + 1], ys[l + 1][j]); ctx.stroke();
        }
      }
    }

    // neurônios
    for (let l = 0; l < tam.length; l++) {
      const a = D ? D[l] : new Array(tam[l]).fill(0);
      let mx = 1e-9, mn = Infinity;
      for (const v of a) { if (v > mx) mx = v; if (v < mn) mn = v; }
      for (let i = 0; i < tam[l]; i++) {
        const x = xs[l], y = ys[l][i], v = a[i];
        if (l === 0) { // entradas: azul (negativo) ↔ laranja (positivo)
          const t = clamp(Math.abs(v) / 3, 0, 1);
          const cor = mix(CINZA, v >= 0 ? LARANJA : AZUL, 0.25 + 0.75 * t);
          ctx.fillStyle = rgba(cor, 1); ctx.strokeStyle = rgba(v >= 0 ? LARANJA : AZUL, 0.5 + 0.5 * t); ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.arc(x, y, 8, 0, 6.2832); ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#c9d1d9'; ctx.textAlign = 'right'; ctx.font = (narrow ? '500 10px' : '500 11px') + ' Inter, system-ui, sans-serif';
          ctx.fillText(NOMES[i], x - 14, y - 6);
          ctx.fillStyle = '#8491a3'; ctx.font = '10px JetBrains Mono, monospace';
          ctx.fillText(f2.format(v), x - 14, y + 7);
        } else if (l === tam.length - 1) { // saída: notas Q
          const rng = (mx - mn) || 1, t = (v - mn) / rng;
          const escolhido = S.ultima && S.ultima.acao === i;
          const cor = mix(CINZA, VERDE, 0.2 + 0.8 * t);
          ctx.save();
          if (escolhido) { ctx.shadowColor = rgba(VERDE, 0.9); ctx.shadowBlur = 18; }
          ctx.fillStyle = rgba(cor, 1); ctx.strokeStyle = escolhido ? '#ffffff' : '#2b3844'; ctx.lineWidth = escolhido ? 2.5 : 1;
          ctx.beginPath(); ctx.arc(x, y, narrow ? 13 : 17, 0, 6.2832); ctx.fill(); ctx.stroke();
          ctx.restore();
          ctx.textAlign = 'left'; ctx.fillStyle = escolhido ? '#ffffff' : '#c9d1d9'; ctx.font = '600 12px Inter, system-ui, sans-serif';
          ctx.fillText(narrow ? nomesAcao[i] : nomesAcao[i] + ' (' + Math.round(S.IA.pesos().niveis[i] * 100) + '%)', x + (narrow ? 22 : 26), y - 7);
          ctx.fillStyle = '#8491a3'; ctx.font = '11px JetBrains Mono, monospace';
          ctx.fillText((narrow ? '' : 'Q = ') + f3.format(v), x + (narrow ? 22 : 26), y + 8);
        } else { // ocultas: brilho = ativação
          const t = clamp(v / mx, 0, 1);
          if (v <= 0) {
            ctx.fillStyle = '#1b242b'; ctx.strokeStyle = '#2b3844'; ctx.lineWidth = 0.6;
            ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 6.2832); ctx.fill(); ctx.stroke();
          } else {
            ctx.save();
            if (t > 0.55) { ctx.shadowColor = rgba(VERDE, 0.85); ctx.shadowBlur = 4 + 8 * t; }
            ctx.fillStyle = rgba(mix(CINZA, VERDE, 0.25 + 0.75 * t), 1);
            ctx.beginPath(); ctx.arc(x, y, 2.6 + 1.9 * t, 0, 6.2832); ctx.fill();
            ctx.restore();
          }
        }
      }
    }
  }

  /* ── painéis de texto ───────────────────────────────────────── */
  function barra(pct, cor, centro) {
    const p = clamp(pct, 0, 100);
    return centro
      ? `<div class="relative h-1.5 rounded bg-[#151b20] flex-1"><div class="absolute top-0 bottom-0 w-px bg-[#2b3844]" style="left:50%"></div>
         <div class="absolute top-0 bottom-0 rounded" style="background:${cor};${pct >= 0 ? `left:50%;width:${Math.abs(pct) / 2}%` : `right:50%;width:${Math.abs(pct) / 2}%`}"></div></div>`
      : `<div class="h-1.5 rounded bg-[#151b20] flex-1"><div class="h-1.5 rounded" style="background:${cor};width:${p}%"></div></div>`;
  }

  function atualizarTextos() {
    const u = S.ultima;
    const niveis = S.IA.pesos().niveis;
    const nomesAcao = niveis.map((n) => (n === 0 ? 'Fora' : n === 1 ? 'Tudo' : 'Meio') + ' (' + Math.round(n * 100) + '%)');
    if (!u) {
      $('#rn-decisao').textContent = 'Ainda sem decisão: a rede precisa de 21 dias de histórico.';
      return;
    }
    const mantem = Math.abs(u.alvo - u.expo) < 0.02;
    $('#rn-decisao').innerHTML = `Escolheu <b class="text-up">${nomesAcao[u.acao]}</b>. Estava com ${f0.format(u.expo * 100)}% em udiCOIN` +
      (mantem ? ' e <b>mantém</b> a posição (dentro da banda de 2%).' : ` e vai para <b>${f0.format(u.alvo * 100)}%</b>.`);
    $('#rn-date').textContent = u.data.slice(8, 10) + '/' + u.data.slice(5, 7) + '/' + u.data.slice(0, 4);

    $('#rn-feat').innerHTML = u.features.map((v, i) => {
      const pos = i === 5 || i === 4 ? v / (i === 5 ? 1 : 3) * 100 : v / 3 * 100;
      const cor = i >= 5 ? '#b388ff' : v >= 0 ? '#ff8a65' : '#64b5f6';
      return `<div class="flex items-center gap-2" title="${esc(DICAS[i])}">
        <span class="text-xs w-24 shrink-0 text-mute">${NOMES[i]}</span>
        ${barra(i === 4 || i === 5 ? pos : pos, cor, !(i === 4 || i === 5))}
        <span class="num text-xs w-12 text-right">${f2.format(v)}</span></div>`;
    }).join('');

    const mn = Math.min(...u.notas), mx = Math.max(...u.notas), rg = (mx - mn) || 1;
    $('#rn-notas').innerHTML = u.notas.map((v, i) => {
      const esc_ = i === u.acao;
      return `<div class="flex items-center gap-2">
        <span class="text-xs w-24 shrink-0 ${esc_ ? 'text-up font-semibold' : 'text-mute'}">${nomesAcao[i]}</span>
        ${barra(10 + 90 * (v - mn) / rg, esc_ ? '#00e676' : '#3a4a57', false)}
        <span class="num text-xs w-14 text-right ${esc_ ? 'text-up' : ''}">${f3.format(v)}</span></div>`;
    }).join('');

    if (S.target) {
      const a1 = S.target[1], a2 = S.target[2];
      const lig = (a) => a.filter((v) => v > 0).length;
      const linha = (nome, n, t) => `<div class="flex items-center gap-2"><span class="text-xs w-24 shrink-0 text-mute">${nome}</span>${barra(100 * n / t, '#00e676', false)}<span class="num text-xs w-14 text-right">${n}/${t}</span></div>`;
      $('#rn-ativos').innerHTML = linha('Oculta 1', lig(a1), a1.length) + linha('Oculta 2', lig(a2), a2.length);
    }
    const tot = S.acoes.reduce((s, x) => s + x, 0) || 1;
    $('#rn-dist').innerHTML = S.acoes.map((n, i) => `<div class="flex items-center gap-2"><span class="text-xs w-24 shrink-0 text-mute">${nomesAcao[i]}</span>${barra(100 * n / tot, '#00e676', false)}<span class="num text-xs w-14 text-right">${f0.format(100 * n / tot)}%</span></div>`).join('')
      + `<div class="text-[11px] text-mute">${f0.format(tot)} decisões desde que o bot existe.</div>`;
  }

  function textosEstaticos() {
    const info = S.IA.info(), P = S.IA.pesos(), T = P.treino || {};
    const t = info.tamanhos;
    $('#rn-arq').innerHTML = `
      <div class="num text-ink">${t[0]} → ${t[1]} → ${t[2]} → ${t[3]}</div>
      <div class="text-xs text-mute">entrada → oculta 1 (ReLU) → oculta 2 (ReLU) → saída (linear)</div>
      <div class="text-xs text-mute mt-2">Parâmetros (pesos + vieses)</div><div class="num">${f0.format(info.parametros)}</div>
      <div class="text-xs text-mute mt-2">Entradas</div><div class="text-xs">retornos de 1, 5 e 20 dias, razão de médias, volatilidade, exposição e lucro da posição</div>
      <div class="text-xs text-mute mt-2">Saídas</div><div class="text-xs">uma nota por exposição-alvo: ${P.niveis.map((n) => Math.round(n * 100) + '%').join(', ')}</div>`;
    const linhas = [
      ['Algoritmo', T.algoritmo], ['Episódios', T.episodios ? f0.format(T.episodios) + ' de ' + (T.dias_por_episodio || 252) + ' dias' : null],
      ['Mercado do treino', 'tendências ' + P.nivel + '×'], ['Período', T.periodo_treino],
      ['Recompensa', T.recompensa], ['Desconto (γ)', T.gama], ['Taxa de aprendizado', T.taxa_aprendizado],
      ['Lote', T.lote], ['Memória (replay)', T.memoria ? f0.format(T.memoria) + ' jogadas' : null],
      ['Rede alvo', T.rede_alvo_a_cada ? 'atualizada a cada ' + f0.format(T.rede_alvo_a_cada) + ' passos' : null],
      ['Exploração (ε)', T.epsilon_inicial != null ? T.epsilon_inicial + ' → ' + T.epsilon_final : null],
    ].filter((r) => r[1] != null && r[1] !== '');
    $('#rn-treino').innerHTML = linhas.map((r) => `<div class="flex justify-between gap-3 py-1.5 border-b border-[#141a20]"><dt class="text-mute text-xs shrink-0">${r[0]}</dt><dd class="text-xs text-right">${esc(r[1])}</dd></div>`).join('');
  }

  function desenharCurva() {
    const curva = (S.IA.pesos().curva || []);
    const msg = $('#rn-curva-msg');
    if (!curva.length || !global.Chart) {
      msg.classList.remove('hidden');
      msg.textContent = 'Sem curva de treino nos dados exportados. Rode o treino longo e exporte de novo (python -m udicoin.agente.exportar_dqn).';
      return;
    }
    if (S.chart) return;
    const ctx = $('#rn-curva').getContext('2d');
    S.chart = new global.Chart(ctx, {
      type: 'line',
      data: {
        labels: curva.map((c) => f0.format(c.episodio)),
        datasets: [
          { label: 'Rede (validação, %)', data: curva.map((c) => c.val_pct), borderColor: '#00e676', backgroundColor: 'rgba(0,230,118,.1)', tension: 0.25, pointRadius: 3, fill: true },
          { label: 'Comprar e segurar (%)', data: curva.map((c) => c.comprar_segurar), borderColor: '#8491a3', borderDash: [5, 4], pointRadius: 0 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { labels: { color: '#8491a3', boxWidth: 12 } }, tooltip: { callbacks: { label: (c) => c.dataset.label + ': ' + f2.format(c.parsed.y) + '%' } } },
        scales: {
          x: { ticks: { color: '#8491a3' }, grid: { color: '#141a20' }, title: { display: true, text: 'episódios de treino', color: '#8491a3' } },
          y: { ticks: { color: '#8491a3', callback: (v) => v + '%' }, grid: { color: '#141a20' } },
        },
      },
    });
  }

  function desenharHeat() {
    const cv = $('#rn-heat');
    const dpr = global.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (!W) return;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const Wm = S.IA.pesos().camadas[0].W; // [neurônio][entrada]
    const n = Wm.length, k = Wm[0].length;
    let mx = 1e-9;
    Wm.forEach((r) => r.forEach((v) => { if (Math.abs(v) > mx) mx = Math.abs(v); }));
    const padL = 96, cw = (W - padL) / n, ch = (H - 4) / k;
    ctx.font = '10px Inter, system-ui, sans-serif'; ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
    for (let i = 0; i < k; i++) {
      ctx.fillStyle = '#8491a3'; ctx.fillText(NOMES[i], padL - 6, 2 + ch * (i + 0.5));
      for (let j = 0; j < n; j++) {
        const v = Wm[j][i] / mx;
        ctx.fillStyle = rgba(v >= 0 ? VERDE : VERM, 0.06 + 0.94 * Math.abs(v));
        ctx.fillRect(padL + j * cw, 2 + i * ch, Math.max(1, cw - 0.6), Math.max(1, ch - 1));
      }
    }
  }

  /* ── laço de animação ───────────────────────────────────────── */
  function frame() {
    S.raf = 0;
    if (!S.visible) return;
    let mexeu = S.dirty;
    if (S.target && S.disp) {
      for (let l = 0; l < S.target.length; l++) {
        const t = S.target[l], d = S.disp[l];
        for (let i = 0; i < t.length; i++) {
          const dif = t[i] - d[i];
          if (Math.abs(dif) > 1e-4) { d[i] += dif * 0.22; mexeu = true; } else d[i] = t[i];
        }
      }
    }
    if (mexeu) { desenhar(); if (S.dirty) { atualizarTextos(); S.dirty = false; } }
    S.raf = global.requestAnimationFrame(frame);
  }

  /* ── navegação ──────────────────────────────────────────────── */
  function mostrar(nome) {
    const rede = nome === 'rede';
    $('#view-dash').classList.toggle('hidden', rede);
    $('#view-rede').classList.toggle('hidden', !rede);
    ['#nav-dash', '#m-nav-dash'].forEach((s) => { const e = $(s); if (e) { e.classList.toggle('active', !rede); if (s === '#nav-dash') e.toggleAttribute('aria-current', !rede); } });
    ['#nav-rede', '#m-nav-rede'].forEach((s) => { const e = $(s); if (e) e.classList.toggle('active', rede); });
    S.visible = rede;
    if (rede) {
      if (!S.IA) { mostrarAviso(); return; }
      popularSelect();
      S.ultima = null; S.target = null; // força reler o estado atual
      lerEstado();
      if (!S.estatico) { textosEstaticos(); S.estatico = true; }
      desenharCurva();
      S.dirty = true;
      if (!S.raf) S.raf = global.requestAnimationFrame(() => { desenharHeat(); frame(); });
      sincronizarControles();
    } else if (global.location.hash === '#rede') {
      global.history.replaceState(null, '', global.location.pathname + global.location.search);
    }
  }

  function mostrarAviso() {
    const el = $('#rn-aviso');
    el.classList.remove('hidden');
    el.innerHTML = 'A rede ainda não foi carregada. Gere o arquivo com <code class="num">python -m udicoin.agente.exportar_dqn --nivel 5</code> e recarregue a página.';
    $('#rn-canvas').style.display = 'none';
  }

  function sincronizarControles() {
    const st = S.sim.getState();
    $('#rn-play').textContent = st.playing ? 'Pause' : 'Play';
    if (S.setActive) S.setActive($('#rn-speed'), 'speed', st.speed);
  }

  function ligarControles() {
    $('#rn-play').addEventListener('click', () => { S.sim.toggle(); sincronizarControles(); });
    $('#rn-step').addEventListener('click', () => S.sim.advance(1));
    $('#rn-speed').addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      S.sim.setSpeed(Number(b.dataset.speed)); sincronizarControles();
    });
    $('#rn-bot').addEventListener('change', () => { S.botId = Number($('#rn-bot').value); S.ultima = null; S.target = null; S.disp = null; lerEstado(); S.dirty = true; });
    S.sim.on('state', () => { if (S.visible) sincronizarControles(); });
    global.addEventListener('resize', () => { if (S.visible) { S.dirty = true; desenharHeat(); } });
    $('#nav-dash').addEventListener('click', (e) => { e.preventDefault(); mostrar('dash'); });
    $('#nav-rede').addEventListener('click', (e) => { e.preventDefault(); mostrar('rede'); });
    $('#m-nav-dash').addEventListener('click', () => mostrar('dash'));
    $('#m-nav-rede').addEventListener('click', () => mostrar('rede'));
  }

  function init({ sim, IA, setActive }) {
    if (S.inited) return;
    S.sim = sim; S.IA = IA; S.setActive = setActive;
    montarHTML();
    ligarControles();
    S.inited = true;
    sim.on('update', onUpdate);
    if (global.location.hash === '#rede') mostrar('rede');
  }

  global.RedeView = { init, mostrar };
})(window);
