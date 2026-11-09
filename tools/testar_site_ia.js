// Teste do agente de IA no site, sem navegador:  node tools/testar_site_ia.js [caminho/agente_dqn.js]
// 1) confere a rede em JS contra o PyTorch;  2) roda o provedor com um bot de IA por 600 dias e confere a contabilidade.
const fs = require('fs'), path = require('path'), vm = require('vm');
const raiz = path.resolve(__dirname, '..');
const arqPesos = process.argv[2] || path.join(raiz, 'web/data/agente_dqn.js');
global.window = global;
vm.runInThisContext(fs.readFileSync(arqPesos, 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(raiz, 'web/js/agente-ia.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(raiz, 'web/js/data-provider.js'), 'utf8'));
let falhas = 0;
const ok = (c, msg) => { console.log((c ? '✓ ' : '✗ ') + msg); if (!c) falhas++; };

const t = AgenteIA.autoteste();
ok(t.ok, `rede em JS = PyTorch (dif features ${t.dif_features.toExponential(1)}, dif notas ${t.dif_notas.toExponential(1)}, mesma ação: ${t.mesma_acao})`);

// ativações por camada (usadas pela tela "Rede neural") têm de terminar nas mesmas notas da rede
const x0 = AGENTE_DQN.testes[0].features;
const at = AgenteIA.ativacoes(x0), fw = AgenteIA.forward(x0);
ok(at.length === 4 && at[3].every((v, i) => Math.abs(v - fw[i]) < 1e-9) && AgenteIA.info().parametros > 0, `ativacoes() = forward() e info(): ${AgenteIA.info().tamanhos.join('→')}, ${AgenteIA.info().parametros} parâmetros`);

// dólar sintético só para o teste
let s = 12345; const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
let v = 3.0; const rows = [];
for (let i = 0; i < 4000; i++) { v *= Math.exp((rnd() - 0.5) * 0.02); const d = new Date(Date.UTC(2010, 0, 4) + i * 86400000); rows.push([d.toISOString().slice(0, 10), v]); }
const sim = UdiProvider.create({ seed: 7, mode: 'anchor', anchorData: rows, trendMult: AgenteIA.nivel() || 1 });
const ia = sim.getBots().find((b) => b.perfil === 'ia');
ok(!!ia, 'o bot "Rede" (IA) foi criado junto com os outros');
sim.advance(600);
const b = sim.getBots().find((x) => x.perfil === 'ia');
const serie = sim.getSeries(b.id);
const expo = serie.map((p) => p.pos / p.equity);
ok(serie.every((p) => isFinite(p.equity) && p.equity > 0), 'patrimônio sempre finito e positivo');
ok(expo.every((e) => e >= -1e-9 && e <= 1 + 1e-9), `exposição sempre entre 0 e 1 (max ${Math.max(...expo).toFixed(3)})`);
ok(b.caixa >= -1e-6 && b.unidades >= 0, 'caixa e unidades nunca ficam negativos');
const iaSt = sim.getIAState(b.id);
ok(iaSt && iaSt.ultima && iaSt.ultima.notas.length === 3 && iaSt.acoes.reduce((a, c) => a + c, 0) > 0, 'getIAState() traz a última decisão e a contagem de ações');
const ops = sim.getOperations(20000).filter((o) => o.botId === b.id);
ok(ops.length > 0, `o bot operou (${ops.length} operações em 600 dias)`);
ok(ops.every((o) => /rede neural/.test(o.motivo)), 'cada ordem traz o motivo "rede neural: alvo X%"');
console.log(`  lucro do bot IA: ${b.lucro_pct.toFixed(2)}%, exposição média ${(expo.reduce((a, c) => a + c, 0) / expo.length).toFixed(2)}`);
process.exit(falhas ? 1 : 0);
