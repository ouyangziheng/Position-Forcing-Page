/* =====================================================================
   Position Forcing — interactive 3D case gallery
   ---------------------------------------------------------------------
   Renders the bookmarked GLB results with normal shading (RGB = surface
   normal direction), which is the convention used for geometry-only 3D
   papers: it exposes surface detail and artifacts that a textured or
   flat-grey render would hide.

   Cards render lazily — a mesh only downloads once its card scrolls into
   view — and each card gets its own small WebGL renderer. Clicking a card
   opens a larger viewer with orbit controls and shading options.
   ===================================================================== */
import * as THREE from '../vendor/three.module.js';
import { GLTFLoader } from '../vendor/GLTFLoader.js';
import { DRACOLoader } from '../vendor/DRACOLoader.js';
import { OrbitControls } from '../vendor/OrbitControls.js';

const MANIFEST = 'static/cases/manifest.json';

/* ---------- shading modes ------------------------------------------- */
/* Default is a LIT surface, not MeshNormalMaterial.
 *
 * These meshes come from marching-cubes style extraction, so the surface
 * carries genuine high-frequency ripple: measured on the 2.53 M-face case,
 * a vertex normal sits 13.98 deg off its neighbours' at p90. That ripple is
 * in the geometry, not in the pipeline — the shipped GLB matches the source
 * file to 3e-6 per vertex.
 *
 * MeshNormalMaterial writes the normal straight into RGB (n*0.5+0.5), so it
 * shows that ripple at full amplitude. Per-vertex colour jitter against
 * neighbours, same mesh:
 *
 *                        p50    p90    p99   over 8/255
 *   MeshNormalMaterial   0.8   11.0   38.8      13.4%
 *   lit (hemi + key)     0.4    4.9   25.7       6.4%
 *
 * 2.3x more noise at p90, and 13.4% of vertices past the visible threshold —
 * which is the speckle seen on skirts and walls. Lighting integrates the
 * ripple away. Normal map stays available as an explicit toggle, the same
 * arrangement other geometry papers use for their viewers.
 */
function litMaterial() {
  return new THREE.MeshStandardMaterial({
    color: 0xcfc6ba, roughness: 0.55, metalness: 0.0,
    side: THREE.FrontSide, flatShading: false,   /* see normalMaterial above */
  });
}

/* Normal shading — the convention in geometry-only papers.
 *
 * flatShading is deliberately OFF. three.js implements it with screen-space
 * derivatives:
 *     vec3 fdx = dFdx( vViewPosition );
 *     vec3 fdy = dFdy( vViewPosition );
 *     normal   = normalize( cross( fdx, fdy ) );
 * dFdx/dFdy are evaluated per 2x2 pixel quad. These meshes carry 0.8-3.4 M
 * triangles, so at typical viewport sizes a triangle covers about one pixel
 * and the derivative quad straddles several of them — the reconstructed
 * normal becomes noise and the surface renders as dense speckle. Using the
 * interpolated geometry normal (vNormal) instead gives the true orientation
 * at any triangle density. */
function normalMaterial() {
  return new THREE.MeshNormalMaterial({ flatShading: false, side: THREE.FrontSide });
}

function smoothMaterial() {
  return new THREE.MeshStandardMaterial({
    color: 0xcfc6ba, roughness: 0.55, metalness: 0.0,
    side: THREE.FrontSide, flatShading: false,
  });
}

function wireMaterial() {
  return new THREE.MeshBasicMaterial({ color: 0x8a6b5d, wireframe: true });
}

const MATERIALS = {
  lit: litMaterial, normal: normalMaterial,
  smooth: smoothMaterial, wire: wireMaterial,
};

/* ---------- a single mesh viewport ---------------------------------- */
class MeshView {
  constructor(host, opts = {}) {
    this.host = host;
    this.mode = opts.mode || 'normal';
    this.interactive = !!opts.interactive;
    this.spin = opts.spin !== false;
    this.disposed = false;

    const w = Math.max(1, host.clientWidth);
    const h = Math.max(1, host.clientHeight || w);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true, alpha: true, powerPreference: 'low-power',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h, false);
    host.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, w / h, 0.05, 60);
    /* Aim below the model centre so it sits high in frame, clear of the inset
       conditioning image in the bottom-right. Without an explicit lookAt the
       camera aims at its own height and the model drops out of frame at the
       bottom — which cropped the legs off standing figures. At this distance
       the 1.75-unit model has ~0.17 headroom above and ~0.49 below. */
    this.camera.position.set(0, 0.72, 4.5);
    this.target = new THREE.Vector3(0, -0.16, 0);
    this.camera.lookAt(this.target);

    /* Ambient-dominant rig. A single hard key light rakes across the facets and
       turns every few-degree dihedral into a visible step; broad fill keeps the
       form readable while flattening that micro-contrast. Only the lit/smooth
       materials use this — MeshNormalMaterial ignores lighting entirely. */
    this.scene.add(new THREE.HemisphereLight(0xfdfbf6, 0x9a8d7d, 2.0));
    const key = new THREE.DirectionalLight(0xffffff, 0.75);
    key.position.set(1.6, 2.4, 3.0);
    const fill = new THREE.DirectionalLight(0xf2ece2, 0.45);
    fill.position.set(-2.2, 0.6, 1.4);
    const back = new THREE.DirectionalLight(0xe8e2d6, 0.30);
    back.position.set(0.4, -1.2, -2.6);
    this.scene.add(key, fill, back);

    if (this.interactive) {
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.target.copy(this.target);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.enablePan = false;
      this.controls.minDistance = 1.6;
      this.controls.maxDistance = 9;
      this.controls.autoRotate = this.spin;
      this.controls.autoRotateSpeed = 0.575;
      this.controls.update();
    }

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this._angle = 0;
  }

  setMesh(geometry) {
    this.clear();
    /* Geometry normals are required now that nothing relies on screen-space
       derivatives. Draco ships positions only, so compute them here. */
    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    const bb = geometry.boundingBox;
    const centre = bb.getCenter(new THREE.Vector3());
    const size = bb.getSize(new THREE.Vector3());
    const extent = Math.max(size.x, size.y, size.z) || 1;

    this.mesh = new THREE.Mesh(geometry, MATERIALS[this.mode]());
    this.mesh.position.sub(centre);
    this.root.scale.setScalar(1.75 / extent);
    this.root.add(this.mesh);
    this.ready = true;
  }

  setMode(mode) {
    if (!this.mesh || !MATERIALS[mode]) return;
    this.mode = mode;
    const old = this.mesh.material;
    this.mesh.material = MATERIALS[mode]();
    if (old && old.dispose) old.dispose();
  }

  setSpin(on) {
    this.spin = on;
    if (this.controls) this.controls.autoRotate = on;
  }

  frame(dt) {
    if (this.disposed || !this.ready) return;
    if (this.controls) {
      this.controls.update();
    } else if (this.spin) {
      this._angle += dt * 0.00017;
      this.root.rotation.y = this._angle;
    }
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    if (this.disposed) return;
    const w = Math.max(1, this.host.clientWidth);
    const h = Math.max(1, this.host.clientHeight || w);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  clear() {
    if (this.mesh) {
      this.root.remove(this.mesh);
      if (this.mesh.geometry) this.mesh.geometry.dispose();
      if (this.mesh.material) this.mesh.material.dispose();
      this.mesh = null;
    }
    this.ready = false;
  }

  dispose() {
    this.clear();
    this.disposed = true;
    if (this.controls) this.controls.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}

function hasWebGL() {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch (e) {
    return false;
  }
}

/* ---------- geometry loading + cache -------------------------------- */
/* Case meshes are Draco-compressed (500k faces at ~0.7 MB each), so the
   decoder has to be wired in before any of them will parse. The path is
   resolved against this module's own URL, so it keeps working when the site
   is served from a subdirectory (GitHub Pages project sites) rather than a
   domain root. DRACOLoader fetches the .wasm as an arraybuffer, so a host
   that serves it with the wrong MIME type is not a problem. */
const DRACO_PATH = new URL('../vendor/draco/', import.meta.url).href;
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(DRACO_PATH);
const loader = new GLTFLoader();
loader.setDRACOLoader(dracoLoader);
const geoCache = new Map();

function loadGeometry(url) {
  if (geoCache.has(url)) return Promise.resolve(geoCache.get(url).clone());
  return new Promise((resolve, reject) => {
    loader.load(url, (gltf) => {
      let found = null;
      gltf.scene.traverse((o) => { if (!found && o.isMesh) found = o; });
      if (!found) { reject(new Error('no mesh in ' + url)); return; }

      /* Draco decodes to float32 positions, so the transform can be baked
         directly. (An int16 attribute would truncate under applyMatrix4 and
         collapse the mesh to a point — not a concern here, but the guard
         below keeps that from silently regressing.) */
      found.updateWorldMatrix(true, false);
      const geo = found.geometry.clone();
      const src = geo.getAttribute('position');
      if (!(src.array instanceof Float32Array)) {
        const f32 = new Float32Array(src.count * 3);
        for (let i = 0; i < src.count; i++) {
          f32[i * 3] = src.getX(i);
          f32[i * 3 + 1] = src.getY(i);
          f32[i * 3 + 2] = src.getZ(i);
        }
        geo.setAttribute('position', new THREE.BufferAttribute(f32, 3));
      }
      geo.applyMatrix4(found.matrixWorld);
      geo.deleteAttribute('uv');
      geo.deleteAttribute('color');
      geo.deleteAttribute('normal');       /* recomputed per-view */
      /* Cache the 500k grid meshes; the full-resolution one the lightbox
         pulls is far larger and is only ever shown one at a time. */
      if (url.includes('.preview.') && geoCache.size < 14) geoCache.set(url, geo.clone());
      resolve(geo);
    }, undefined, reject);
  });
}

/* ---------- render loop shared by every live viewport ---------------- */
const live = new Set();
let last = 0;
function loop(ts) {
  const dt = last ? Math.min(60, ts - last) : 16;
  last = ts;
  live.forEach((v) => v.frame(dt));
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

/* ---------- build the gallery ---------------------------------------- */
const galEl = document.getElementById('gal');
const barEl = document.getElementById('gal-bar');
const countEl = document.getElementById('gal-count');
const moreWrap = document.getElementById('gal-pager');
if (galEl) boot();

async function boot() {
  /* WebGL is required; say so plainly rather than rendering empty cards. */
  if (!hasWebGL()) {
    galEl.innerHTML =
      '<p class="tiny">This browser reports no WebGL support, so the interactive meshes ' +
      'cannot render. The static comparisons further down the page still work.</p>';
    return;
  }

  let data;
  try {
    const r = await fetch(MANIFEST, { cache: 'no-cache' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    data = await r.json();
  } catch (err) {
    galEl.innerHTML =
      '<p class="tiny">Could not load <span class="mono">' + MANIFEST + '</span> (' +
      err.message + '). Run <span class="mono">python3 tools/build_cases.py</span> to ' +
      'generate the case assets.</p>';
    return;
  }

  const cases = data.cases || [];
  if (!cases.length) {
    galEl.innerHTML = '<p class="tiny">Manifest contains no cases.</p>';
    return;
  }

  /* paged: a fixed grid per page, so only a bounded number of WebGL
     contexts and meshes are ever live at once */
  const PER_PAGE = 12;
  const pages = Math.max(1, Math.ceil(cases.length / PER_PAGE));
  let page = 0;

  const prevBtn = document.getElementById('pg-prev');
  const nextBtn = document.getElementById('pg-next');

  function go(p) {
    page = (p + pages) % pages;      /* wrap around both ways */
    draw();
    galEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  prevBtn.addEventListener('click', () => go(page - 1));
  nextBtn.addEventListener('click', () => go(page + 1));

  /* page dots */
  const dots = [];
  for (let i = 0; i < pages; i++) {
    const d = document.createElement('button');
    d.className = 'pg-dot';
    d.textContent = String(i + 1);
    d.addEventListener('click', () => go(i));
    moreWrap.appendChild(d);
    dots.push(d);
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'PageDown') go(page + 1);
    if (e.key === 'PageUp') go(page - 1);
  });

  /* lazily attach a viewer once a card is visible */
  const io = new IntersectionObserver((ents) => {
    ents.forEach((en) => {
      if (!en.isIntersecting) return;
      io.unobserve(en.target);
      attach(en.target);
    });
  }, { rootMargin: '250px 0px' });

  function attach(card) {
    const host = card.querySelector('.case-view');
    const ph = card.querySelector('.case-ph');
    /* Cards load a 500k preview; the lightbox loads the untouched mesh.
       Full-resolution cards measured 77 MB and 90-115 s for one page of 12
       (vs 8.7 MB / ~19 s), and the decimation only softens edge sharpness —
       it does not terrace flat panels, since reduce_face runs with
       preservenormal/preservetopology. The terracing seen earlier came from
       quantizing at 14 bits, which is now 18. */
    const url = card.dataset.preview || card.dataset.mesh;
    const view = new MeshView(host, { mode: 'normal', interactive: false, spin: true });
    card._view = view;
    loadGeometry(url).then((geo) => {
      if (view.disposed) return;
      view.setMesh(geo);
      live.add(view);
      if (ph) ph.remove();
    }).catch((err) => {
      if (ph) ph.innerHTML = '<span>failed to load</span>';
      console.warn('mesh load failed', url, err);
    });
  }

  function draw() {
    /* tear down existing viewers so WebGL contexts don't leak */
    galEl.querySelectorAll('.case').forEach((c) => {
      if (c._view) { live.delete(c._view); c._view.dispose(); c._view = null; }
    });
    galEl.innerHTML = '';

    const start = page * PER_PAGE;
    const shown = cases.slice(start, start + PER_PAGE);

    /* index into the full list so the lightbox can walk every case */
    shown.forEach((c) => galEl.appendChild(makeCard(c)));
    countEl.textContent = `${start + 1}–${start + shown.length} of ${cases.length}`;

    dots.forEach((d, i) => d.classList.toggle('on', i === page));
    moreWrap.style.display = pages > 1 ? 'flex' : 'none';

    galEl.querySelectorAll('.case').forEach((card) => io.observe(card));
  }

  function makeCard(c) {
    const card = document.createElement('article');
    card.className = 'case';
    card.dataset.mesh = c.mesh;
    if (c.preview) card.dataset.preview = c.preview;
    card.innerHTML =
      '<div class="case-view">' +
        '<div class="case-ph"><div class="spin"></div><span>loading mesh</span></div>' +
        '<div class="case-in"><img src="' + c.image + '" alt="conditioning image" loading="lazy"></div>' +
      '</div>';
    return card;
  }

  draw();
  window.addEventListener('resize', () => {
    galEl.querySelectorAll('.case').forEach((c) => { if (c._view) c._view.resize(); });
  });
}
