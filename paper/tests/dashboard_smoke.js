// Stub-DOM smoke run of docs/design/dexposure-claw-case-workbench/index.html against the real assets.
// Prints JSON {bad: [...], kpis: {...}}. Used by test_dashboard_contagion.py.
const fs = require('fs'), path = require('path');
const WB = process.argv[2];
const html = fs.readFileSync(path.join(WB, 'index.html'), 'utf8');
const js = html.split('<script>')[1].split('</script>')[0];
global.window = global;
require(path.join(WB, 'assets', 'contagion.js'));
const els = {};
function el(id) { if (!els[id]) els[id] = { id, textContent: '', innerHTML: '', value: '', disabled: false, style: {}, dataset: {},
  classList: { toggle() {}, remove() {} }, appendChild() {}, setAttribute() {}, focus() {},
  options: [], insertAdjacentHTML(_, h) { this.innerHTML = h + this.innerHTML; },
  set outerHTML(v) { this._outer = v; }, get outerHTML() { return this._outer || ''; } }; return els[id]; }
global.document = { documentElement: el('html'), getElementById: el, querySelectorAll: () => [], body: el('body'), createElement: () => el('_tmp_' + Math.random()) };
global.addEventListener = () => {};
global.fetch = async p => ({ ok: true, json: async () => JSON.parse(fs.readFileSync(path.join(WB, p.split('?')[0]), 'utf8')) });
global.THREE = { Vector3: class { constructor(x=0,y=0,z=0){ this.x=x; this.y=y; this.z=z; } } };
eval(js + '\n;global.__f={init,setHorizon,setLens,setSimMode,runSim,simulateTokenDepeg,setEdition,toggleTheme,setUserLens,runUserPreset,setUserLeftMode,setLeftMode,setWfBoard,setBt,compute3dPositions,netH};');
(async () => {
  el('sortSel').value = 'poolDropPct'; el('simShock').value = '50'; el('uSimShock').value = '20';
  await __f.init();
  const bad = [];
  const kpis = {};
  for (const h of ['1', '4', '8']) {
    __f.setHorizon(h);
    const pos3d = __f.compute3dPositions(__f.netH());
    const ySet = new Set(Object.values(pos3d).map(v => v.y));
    for (const expectedY of [108, 38, -34, -106]) {
      if (!ySet.has(expectedY)) bad.push(`3dDeck missing y=${expectedY} h=${h}`);
    }
    __f.setEdition('user');
    __f.setUserLeftMode('quad');
    __f.setUserLeftMode('flow');
    __f.setUserLeftMode('quad');
    for (const ul of ['alpha', 'mid', 'm7', 'hhi']) {
      __f.setUserLens(ul);
      if (/undefined|NaN/.test(el('uListBody').innerHTML)) bad.push(`uListBody h=${h} ulens=${ul}`);
    }
    for (const wb of ['alpha', 'whale']) {
      __f.setWfBoard(wb);
      if (/undefined|NaN/.test(el('uTrackBody').innerHTML) || /undefined|NaN/.test(el('wfBody').innerHTML))
        bad.push(`wfBoard=${wb} h=${h}`);
    }
    for (const pr of ['depeg_usdc_usdt', 'proto_lido', 'proto_top3', 'proto_alpha3']) {
      __f.runUserPreset(pr);
      if (/undefined|NaN/.test(el('uSimSummary').innerHTML) || /undefined|NaN/.test(el('uSimBody').innerHTML))
        bad.push(`userPreset ${pr} h=${h}`);
    }
    for (const uk of ['quadSvg', 'uCards', 'uHealthCard', 'uFlowSvg', 'uEdgeBody', 'uSimKpis', 'uSimCascSvg', 'uGuideCol1', 'uGuideCol2', 'uGuideCol3', 'uTrackKpis', 'uTrackBody', 'uCrisisCards']) {
      if (/undefined|NaN/.test(el(uk).innerHTML) || !el(uk).innerHTML.trim()) bad.push(`${uk} h=${h}`);
    }
    __f.toggleTheme();
    __f.setEdition('pro');
    __f.setBt('v4h');
    if (/undefined|NaN/.test(el('v4hBody').innerHTML) || !el('v4hBody').innerHTML.trim()) bad.push(`v4hBody h=${h}`);
    __f.setBt('wf');
    __f.setLeftMode('quad');
    __f.setLeftMode('2d');
    __f.setSimMode('protocol');
    el('simSel').value = (el('simSel').innerHTML.match(/value="([^"]+)"/) || [])[1];
    for (const l of ['alpha', 'mid', 'm7', 'm2']) {
      __f.setLens(l);
      if (/undefined|NaN/.test(el('rankBody').innerHTML)) bad.push(`rankBody h=${h} lens=${l}`);
    }
    __f.runSim();
    for (const k of ['wfBody', 'terraBody', 'ftxBody', 'p29Body', 'simBody', 'hhiBody', 'gateBody', 'edgeBody', 'dagSvg', 'proQuadSvg', 'cascSvg'])
      if (/undefined|NaN|<td[^>]*>\s*→/.test(el(k).innerHTML) || !el(k).innerHTML.trim()) bad.push(`${k} h=${h}`);
    for (const k of ['kSnap', 'kSnapSub', 'kQueue', 'kQueueSub', 'kEdges', 'kEdgesSub', 'kWf', 'kWfSub', 'k29', 'k29Sub', 'sDirect', 'sObs', 'sPred']) {
      kpis[`${k}@${h}`] = el(k).textContent;
      if (/undefined|NaN/.test(el(k).textContent)) bad.push(`${k} h=${h}`);
    }
    __f.simulateTokenDepeg('USDC');
    kpis[`depegObs@${h}`] = el('sObs').textContent;
    kpis[`depegPred@${h}`] = el('sPred').textContent;
    if (/undefined|NaN/.test(el('simBody').innerHTML) || /undefined|NaN/.test(el('cascSvg').innerHTML))
      bad.push(`depeg h=${h}`);
  }
  process.stdout.write(JSON.stringify({ bad, kpis }));
})().catch(e => { console.error(e); process.exit(1); });
