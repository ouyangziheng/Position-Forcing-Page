/* =====================================================================
   Position Forcing — generation-process viewer
   ---------------------------------------------------------------------
   Walks one 50-step sampling trajectory for a real case, showing the three
   things the method ties together:

     · the conditioning image           (what the model is given)
     · the quantized token positions    (the spatial guidance being fed back)
     · the final mesh                   (what the trajectory converges to)

   The resolution schedule is the paper's Eq. 8 evaluated directly:

       m(t) = floor( t*m_min + (1-t)*m_max )
       R(t) = clip( 2^m(t), R_min, R_max )   m_min=-1, m_max=9
                                             R_min=1,  R_max=128

   Positions come from surface samples of the actual generated asset, so
   quantizing them on an R^3 grid performs the same Q_R mapping the paper
   applies to recovered positions.
   ===================================================================== */
import * as THREE from '../vendor/three.module.js';
import { GLTFLoader } from '../vendor/GLTFLoader.js';
import { DRACOLoader } from '../vendor/DRACOLoader.js';
import { OrbitControls } from '../vendor/OrbitControls.js';

const STEPS = 50;
const M_MIN = -1, M_MAX = 9;
const R_MIN = 1, R_MAX = 128;
const META = 'static/demo/demo.json';

/* ---------- schedule ------------------------------------------------- */
/* Eq. 8 evaluated on the step index rather than on t.
 *
 * With t = (STEPS - i) / STEPS, the exponent
 *     m(t) = t*M_MIN + (1-t)*M_MAX
 * is exactly (M_MAX*STEPS - (M_MAX - M_MIN)*(STEPS - i)) / STEPS, an integer
 * ratio. Computing it in floating point instead lands a hair under the integer
 * at some steps — at i=10, 1-0.8 is 0.19999999999999996, so m comes out
 * 0.9999999999999996 and floors to 0 instead of 1. That shifts one step from
 * R=2 into R=1 and makes the resolution bands unequal, when every interior
 * exponent should span exactly STEPS/(M_MAX - M_MIN) = 5 steps.
 */
function resAtStep(i) {
  const num = M_MAX * STEPS - (M_MAX - M_MIN) * (STEPS - i);
  const m = Math.floor(num / STEPS);
  return Math.min(Math.max(Math.pow(2, m), R_MIN), R_MAX);
}
const SCHED = [];
for (let i = 0; i <= STEPS; i++) {
  SCHED.push({ i, t: (STEPS - i) / STEPS, R: resAtStep(i) });
}

/* The demo mesh is Draco-compressed like the gallery cases. Resolve the
   decoder against this module's URL so a subdirectory deploy still finds it. */
const DRACO_PATH = new URL('../vendor/draco/', import.meta.url).href;

function loadMesh(url) {
  const draco = new DRACOLoader();
  draco.setDecoderPath(DRACO_PATH);
  const loader = new GLTFLoader();
  loader.setDRACOLoader(draco);
  return new Promise((res, rej) => loader.load(url, res, undefined, rej));
}

/* ---------- DOM ------------------------------------------------------ */
const host = document.getElementById('proc-view');
if (host) boot();

async function boot() {
  const elStep = document.getElementById('ro-step');
  const elT = document.getElementById('ro-t');
  const elRes = document.getElementById('ro-res');
  const elStgRes = document.getElementById('stage-res');
  const scrub = document.getElementById('scrub');
  const playBtn = document.getElementById('play');
  const resetBtn = document.getElementById('reset');
  const bandsEl = document.getElementById('bands');
  const flowEls = [1, 2, 3, 4, 5].map((k) => document.getElementById('f' + k));
  const layerBtns = [...document.querySelectorAll('[data-layer]')];

  /* ---------- trajectory diagram ------------------------------------ */
  /* Each sampling phase lights the edges it uses and a pulse travels along
     the one carrying information, so the diagram animates the same loop the
     viewer is stepping through. Mirrors panel (d) of Fig. 1.
     The final phase lights BOTH inbound edges: z_{t-1} is produced from z_t
     by the Euler update *and* conditioned on P_{0|t} through RoPE — the two
     arrive together, which is the whole point of the design. */
  const PHASES = [
    /* Phase 0 only evaluates the velocity AT z_t — nothing has moved to
       z_{t-1} yet, so no edge is traversed. Just mark the node. */
    { edges: [], nodes: ['n-zt'], tracks: [] },
    { edges: ['e-drop'], nodes: ['n-zt', 'n-z0h'], tracks: [[[119, 56], [119, 106]]] },
    { edges: ['e-pos'], nodes: ['n-z0h', 'n-p0t'], tracks: [[[132, 120], [218, 120]]] },
    { edges: ['e-pos'], nodes: ['n-p0t'], tracks: [[[218, 120], [235, 120]]] },
    /* Both inbound edges run together: z_{t-1} comes from the Euler step off
       z_t *and* is conditioned on P_{0|t} through RoPE. Two pulses converge
       on the same node so that simultaneity is visible. */
    {
      edges: ['e-main', 'e-rope'],
      nodes: ['n-zt', 'n-p0t', 'n-ztm1'],
      tracks: [[[131, 44], [224, 44]], [[235, 106], [235, 58]]],
    },
  ];
  const pulses = [document.getElementById('pulse'), document.getElementById('pulse2')];
  const trajSvg = document.getElementById('traj');

  function paintPhase(idx, frac) {
    if (!trajSvg) return;
    trajSvg.querySelectorAll('path.lit').forEach((p) => p.classList.remove('lit'));
    trajSvg.querySelectorAll('.nd.lit').forEach((n) => n.classList.remove('lit'));
    if (idx < 0) { pulses.forEach((p) => p && p.classList.remove('on')); return; }
    const ph = PHASES[idx];
    ph.edges.forEach((id) => {
      const e = document.getElementById(id);
      if (e) e.classList.add('lit');
    });
    ph.nodes.forEach((id) => {
      const n = document.getElementById(id);
      if (n) n.classList.add('lit');
    });
    const u = Math.max(0, Math.min(1, frac));
    pulses.forEach((p, k) => {
      if (!p) return;
      const tr = ph.tracks[k];
      if (!tr) { p.classList.remove('on'); return; }
      const [a, b] = tr;
      p.setAttribute('cx', String(a[0] + (b[0] - a[0]) * u));
      p.setAttribute('cy', String(a[1] + (b[1] - a[1]) * u));
      p.classList.add('on');
    });
  }

  scrub.max = String(STEPS);

  /* resolution bands along the timeline */
  (function bands() {
    const runs = [];
    let cur = null;
    SCHED.forEach((s) => {
      if (!cur || cur.R !== s.R) { cur = { R: s.R, from: s.i, to: s.i }; runs.push(cur); }
      else cur.to = s.i;
    });
    runs.forEach((r) => {
      const d = document.createElement('div');
      d.className = 'band';
      d.dataset.from = r.from; d.dataset.to = r.to;
      d.style.flex = String(r.to - r.from + 1);
      d.textContent = 'R=' + r.R;
      d.title = `steps ${r.from}–${r.to} · R = ${r.R}`;
      bandsEl.appendChild(d);
    });
  })();
  const bandEls = [...bandsEl.children];

  /* ---------- load assets ------------------------------------------- */
  let meta, levels, mesh;
  try {
    meta = await (await fetch(META, { cache: 'no-cache' })).json();
    const [cbuf, gltf] = await Promise.all([
      fetch(meta.cells, { cache: 'no-cache' }).then((r) => r.arrayBuffer()),
      loadMesh(meta.mesh),
    ]);

    /* cells.bin: the mesh is fully voxelized per level (every cell its surface
       crosses), then 6144 of those cells are chosen by farthest-point so they
       cover the whole shape. Binning sampled points in the browser instead left
       gaps — at R=128 the surface occupies ~80k cells and no practical number
       of random samples reaches them all. */
    const dv = new DataView(cbuf);
    if (String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3)) !== 'PFC1') {
      throw new Error('cells.bin: bad magic');
    }
    levels = new Map();
    const nLev = dv.getUint32(4, true);
    let off = 8;
    for (let i = 0; i < nLev; i++) {
      const R = dv.getUint16(off, true);
      const count = dv.getUint32(off + 2, true);
      off += 6;
      levels.set(R, new Uint16Array(cbuf, off, count * 3));
      off += count * 6;
    }

    let found = null;
    gltf.scene.traverse((o) => { if (!found && o.isMesh) found = o; });
    found.updateWorldMatrix(true, false);
    /* widen to Float32 before baking the transform: the quantized GLB stores
       Int16 positions with a ~1e-5 node scale, and applyMatrix4 writes back
       into the source array — an integer array truncates every scaled
       coordinate to zero and the mesh collapses to a point. */
    const geo = found.geometry.clone();
    const src = geo.getAttribute('position');
    if (!(src.array instanceof Float32Array)) {
      const f32 = new Float32Array(src.count * 3);
      for (let i = 0; i < src.count; i++) {
        f32[i * 3] = src.getX(i); f32[i * 3 + 1] = src.getY(i); f32[i * 3 + 2] = src.getZ(i);
      }
      geo.setAttribute('position', new THREE.BufferAttribute(f32, 3));
    }
    geo.applyMatrix4(found.matrixWorld);
    geo.deleteAttribute('uv');
    geo.deleteAttribute('color');
    /* Keep/derive geometry normals: flatShading is off because three.js
       implements it with dFdx/dFdy per 2x2 pixel quad, which turns into noise
       once triangles approach pixel size (this mesh has 800k). */
    if (!geo.getAttribute('normal')) geo.computeVertexNormals();
    mesh = geo;
  } catch (err) {
    host.innerHTML = '<div class="proc-err">Could not load the demo assets (' +
      (err && err.message ? err.message : err) + '). Run <span class="mono">python3 ' +
      'tools/build_demo_tokens.py</span> to generate them.</div>';
    return;
  }

  /* ---------- three.js scene ---------------------------------------- */
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 60);
  /* Pulled back far enough that the R=1/R=2 cells — which span the whole
     normalized box — still sit inside the frame. */
  camera.position.set(2.9, 1.75, 3.5);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minDistance = 2.4;
  controls.maxDistance = 9;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.85;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x6b6257, 0.85));
  const key = new THREE.DirectionalLight(0xffffff, 1.9);
  key.position.set(2.6, 3.4, 2.2);
  const rim = new THREE.DirectionalLight(0xf0e3d4, 0.8);
  rim.position.set(-2.4, 1.2, -2.1);
  scene.add(key, rim);

  /* Final mesh — flat warm stone, semi-transparent, acting as the target the
     trajectory converges to. A single colour keeps it reading as context
     behind the brown position cells rather than competing with them. */
  const meshMat = new THREE.MeshStandardMaterial({
    color: 0xcfc3b4, roughness: 0.74, metalness: 0.0,
    transparent: true, opacity: 0.7, depthWrite: false, side: THREE.FrontSide,
  });
  const meshObj = new THREE.Mesh(mesh, meshMat);
  scene.add(meshObj);

  /* Quantized token positions as instanced boxes. The diagram draws half the
     paper's 6144 tokens — a sparser set reads more clearly — so no level can
     produce more cells than that. */
  const MAX_CELLS = 3072;
  const cellGeo = new THREE.BoxGeometry(1, 1, 1);
  const cellMat = new THREE.MeshStandardMaterial({
    color: 0x8a6a4f, roughness: 0.52, metalness: 0.05,
  });
  const cells = new THREE.InstancedMesh(cellGeo, cellMat, MAX_CELLS);
  cells.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  cells.count = 0;
  cells.frustumCulled = false;
  scene.add(cells);

  /* ---------- cell lookup ------------------------------------------- */
  const HALF = (meta.normExtent || 1.8) / 2;
  const cellCache = new Map();
  const dummy = new THREE.Object3D();

  /* Levels are precomputed, so this only turns indices into world centres. */
  function quantize(t, R) {
    const key = 'R' + R;
    if (cellCache.has(key)) return cellCache.get(key);

    const src = levels.get(R);
    const inv = (2 * HALF) / R;
    const n = src ? src.length / 3 : 0;
    const arr = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      arr[i * 3]     = -HALF + (src[i * 3] + 0.5) * inv;
      arr[i * 3 + 1] = -HALF + (src[i * 3 + 1] + 0.5) * inv;
      arr[i * 3 + 2] = -HALF + (src[i * 3 + 2] + 0.5) * inv;
    }
    /* Thinned levels keep only a fraction of the occupied cells, so the cubes
       are grown by sqrt(occupied / kept) to cover the same surface the full
       voxel set did. 1.0 for levels that were not thinned. */
    const grow = (meta.cellScale && meta.cellScale[String(R)]) || 1;
    const out = { xyz: arr, n, size: inv, grow };
    cellCache.set(key, out);
    return out;
  }

  /* ---------- layer toggles ----------------------------------------- */
  const show = { cells: true, mesh: true };

  function applyLayers() {
    meshObj.visible = show.mesh;
    cells.visible = show.cells;
    layerBtns.forEach((b) => b.classList.toggle('on', !!show[b.dataset.layer]));
  }
  layerBtns.forEach((b) => b.addEventListener('click', () => {
    show[b.dataset.layer] = !show[b.dataset.layer];
    applyLayers();
  }));

  /* ---------- state ------------------------------------------------- */
  let cur = 0, playing = false, speed = 1, acc = 0, lastTs = 0;
  let flowPhase = 0, flowAcc = 0;
  const STEP_MS = 330;

  function update() {
    const s = SCHED[cur];
    const q = quantize(s.t, s.R);

    /* Draw every quantized cell as a box at its exact cell size. An earlier
       1.45x inflation at fine resolutions closed the gaps between sparse
       cells, but it pushed the cubes out past the surface they were sampled
       from, so they no longer looked like they sat on the mesh. */
    const n = Math.min(q.n, MAX_CELLS);
    const sz = q.size * q.grow;
    for (let i = 0; i < n; i++) {
      dummy.position.set(q.xyz[i * 3], q.xyz[i * 3 + 1], q.xyz[i * 3 + 2]);
      dummy.scale.setScalar(sz);
      dummy.updateMatrix();
      cells.setMatrixAt(i, dummy.matrix);
    }
    cells.count = n;
    cells.instanceMatrix.needsUpdate = true;
    cells.visible = show.cells;


    /* The mesh firms up as the trajectory converges, but stays translucent
       enough that the cells in front of it remain readable. */
    meshObj.material.opacity = 0.42 + 0.33 * (1 - s.t);

    elStep.textContent = s.i + ' / ' + STEPS;
    elT.textContent = s.t.toFixed(3);
    elRes.textContent = String(s.R);
    elStgRes.textContent = 'R = ' + s.R;
    if (scrub.value !== String(cur)) scrub.value = String(cur);

    bandEls.forEach((b) => b.classList.toggle('on', cur >= +b.dataset.from && cur <= +b.dataset.to));
    flowEls.forEach((f, k) => { if (f) f.classList.toggle('on', k === flowPhase && playing); });
  }

  /* The stage column stretches to match the diagram column, so take the
     height from the laid-out box rather than forcing an aspect ratio —
     otherwise the canvas and its container disagree and the render is
     letterboxed or stretched. A minimum keeps it usable on narrow screens. */
  function resize() {
    const w = Math.max(1, host.clientWidth);
    const h = Math.max(260, host.clientHeight || Math.round(w * 0.72));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function tick(ts) {
    const dt = lastTs ? Math.min(64, ts - lastTs) : 16;
    lastTs = ts;
    if (playing) {
      acc += dt * speed;
      flowAcc += dt * speed;
      /* The last phase holds longer: it is the one where the Euler update and
         the RoPE injection converge on z_{t-1}, and at 460 ms that reads as a
         flicker rather than as two things arriving together. */
      const PHASE_MS = flowPhase === 4 ? 900 : 460;
      if (flowAcc > PHASE_MS) {
        flowAcc -= PHASE_MS;
        flowPhase = (flowPhase + 1) % 5;
        flowEls.forEach((f, k) => { if (f) f.classList.toggle('on', k === flowPhase); });
      }
      paintPhase(flowPhase, flowAcc / PHASE_MS);
      if (acc >= STEP_MS) {
        acc = 0;
        if (cur >= STEPS) setPlaying(false);
        else { cur++; update(); }
      }
    }
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }

  function setPlaying(p) {
    playing = p;
    playBtn.textContent = p ? '❚❚ Pause' : (cur >= STEPS ? '↻ Replay' : '▶ Play');
    if (!p) {
      flowEls.forEach((f) => { if (f) f.classList.remove('on'); });
      paintPhase(-1, 0);            /* clear the diagram when idle */
    }
    update();
  }

  playBtn.addEventListener('click', () => {
    if (cur >= STEPS) { cur = 0; acc = 0; }
    setPlaying(!playing);
  });
  resetBtn.addEventListener('click', () => { cur = 0; acc = 0; setPlaying(false); });
  scrub.addEventListener('input', () => {
    cur = +scrub.value; acc = 0;
    if (playing) setPlaying(false);
    update();
  });
  document.querySelectorAll('.spd-btn').forEach((b) => {
    b.addEventListener('click', () => {
      speed = parseFloat(b.dataset.spd);
      document.querySelectorAll('.spd-btn').forEach((o) => o.classList.remove('is-on'));
      b.classList.add('is-on');
    });
  });
  const spinBox = document.getElementById('spin');
  if (spinBox) spinBox.addEventListener('change', () => { controls.autoRotate = spinBox.checked; });

  window.addEventListener('resize', () => { resize(); });

  resize();
  applyLayers();
  update();
  requestAnimationFrame(tick);

  if ('IntersectionObserver' in window) {
    new IntersectionObserver((ents) => {
      ents.forEach((en) => { if (en.isIntersecting) controls.autoRotate = true; });
    }, { threshold: 0.4 }).observe(document.getElementById('process'));
  }
}
