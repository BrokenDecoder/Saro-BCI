/**
 * Brain3D — Premium WebGL Neural Topography Renderer
 * ====================================================
 * Features:
 *  - Semi-transparent glass skull (outer) + textured brain inner sphere
 *  - 128 glowing electrode nodes on a golden-spiral scalp layout
 *  - Medical heatmap: cold-blue → teal → green → amber → hot-red
 *  - Dynamic neural "synapse" connection mesh between top active nodes
 *  - Animated particle sparks that fire along active connections
 *  - Shockwave ring pulse on anomaly alert with zone localization
 *  - Smooth 20ms interpolation for all power values
 *  - Drag to rotate, scroll to zoom (OrbitControls)
 */
class Brain3D {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container || typeof THREE === 'undefined') return;

    this.nChannels   = 128;
    this.nodes       = [];             // electrode node objects
    this.sparks      = [];             // particle sparks
    this.connections = null;           // THREE.LineSegments
    this.shockwaves  = [];             // expanding ring meshes
    this.targetPowers  = new Float32Array(this.nChannels);
    this.currentPowers = new Float32Array(this.nChannels);
    this.smoothPowers  = new Float32Array(this.nChannels);
    this.alertZone   = null;
    this.alertDecay  = 0;
    this.time        = 0;
    this.autoRotate  = true;

    // Rolling EMA max for stable heatmap colour scale (no per-frame flicker)
    this._emaMax     = 1e-6;
    this._EMA_ALPHA  = 0.03;  // slow decay → stable over ~10s at 1 Hz updates

    // Color stops for medical EEG heatmap (fMRI-style)
    this.colorMap = [
      new THREE.Color(0x0d1b4b), // 0.0 – deep navy (silent)
      new THREE.Color(0x1a56a6), // 0.2 – blue
      new THREE.Color(0x06b6d4), // 0.4 – cyan
      new THREE.Color(0x10b981), // 0.6 – green
      new THREE.Color(0xf59e0b), // 0.8 – amber
      new THREE.Color(0xef4444), // 1.0 – red (max)
    ];

    this._init();
    this._buildHead();
    this._buildElectrodes();
    this._buildConnectionMesh();
    this._buildLegend();
    this._animate();

    window.addEventListener('resize', () => this._onResize());
  }

  /* ── Scene Setup ─────────────────────────────────────────────────── */
  _init() {
    const W = this.container.clientWidth  || 600;
    const H = this.container.clientHeight || 400;

    this.scene = new THREE.Scene();

    // Camera
    this.camera = new THREE.PerspectiveCamera(42, W / H, 0.1, 500);
    this.camera.position.set(0, 2, 9);
    this.camera.lookAt(0, 0, 0);

    // Renderer — transparent background
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(W, H);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.container.appendChild(this.renderer.domElement);

    // OrbitControls
    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping  = true;
      this.controls.dampingFactor  = 0.07;
      this.controls.enablePan      = false;
      this.controls.minDistance    = 4;
      this.controls.maxDistance    = 18;
      this.controls.autoRotate     = true;
      this.controls.autoRotateSpeed = 0.6;

      // Stop auto-rotate while user is dragging; resume after 3s idle
      this._autoRotateTimer = null;
      this.renderer.domElement.addEventListener('pointerdown', () => {
        if (this.controls) this.controls.autoRotate = false;
        clearTimeout(this._autoRotateTimer);
      });
      this.renderer.domElement.addEventListener('pointerup', () => {
        clearTimeout(this._autoRotateTimer);
        this._autoRotateTimer = setTimeout(() => {
          if (this.controls) this.controls.autoRotate = true;
        }, 3000);
      });
    }

    // Lights
    this.scene.add(new THREE.AmbientLight(0x334466, 1.5));

    const rim1 = new THREE.DirectionalLight(0x06b6d4, 1.2);
    rim1.position.set(-4, 6, -3);
    this.scene.add(rim1);

    const rim2 = new THREE.DirectionalLight(0x8b5cf6, 0.7);
    rim2.position.set(5, -2, 4);
    this.scene.add(rim2);

    const topLight = new THREE.PointLight(0xffffff, 0.5, 30);
    topLight.position.set(0, 8, 0);
    this.scene.add(topLight);
  }

  /* ── Head Geometry ──────────────────────────────────────────────── */
  _buildHead() {
    // Inner brain — opaque with subtle texture
    const brainGeo = new THREE.SphereGeometry(2.2, 48, 48);
    brainGeo.applyMatrix4(new THREE.Matrix4().makeScale(1.0, 1.15, 1.05));

    const brainMat = new THREE.MeshPhongMaterial({
      color:     0x0d1b3e,
      emissive:  0x060d20,
      shininess: 30,
      transparent: true,
      opacity: 0.92,
      side: THREE.FrontSide,
    });
    this.brainMesh = new THREE.Mesh(brainGeo, brainMat);
    this.scene.add(this.brainMesh);

    // Outer skull — glassy wireframe shell
    const skullGeo = new THREE.SphereGeometry(2.85, 36, 36);
    skullGeo.applyMatrix4(new THREE.Matrix4().makeScale(1.0, 1.15, 1.05));

    const skullMat = new THREE.MeshPhongMaterial({
      color:       0x1a3a6b,
      emissive:    0x0a1f40,
      transparent: true,
      opacity:     0.12,
      wireframe:   false,
      side:        THREE.FrontSide,
      shininess:   120,
    });
    this.skullMesh = new THREE.Mesh(skullGeo, skullMat);
    this.scene.add(this.skullMesh);

    // Skull wireframe overlay for premium feel
    const wireGeo = new THREE.SphereGeometry(2.87, 20, 20);
    wireGeo.applyMatrix4(new THREE.Matrix4().makeScale(1.0, 1.15, 1.05));
    const wireMat = new THREE.MeshBasicMaterial({
      color:       0x1e4080,
      wireframe:   true,
      transparent: true,
      opacity:     0.10,
    });
    this.scene.add(new THREE.Mesh(wireGeo, wireMat));

    // Equator ring for anatomical reference
    this._addRing(3.0, 0.04, 0x1e4080, 0.35);
    // Sagittal midline arc (faint)
    this._addArc();
  }

  _addRing(radius, tubeR, color, opacity) {
    const geo = new THREE.TorusGeometry(radius, tubeR, 6, 64);
    const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity });
    const m = new THREE.Mesh(geo, mat);
    m.rotation.x = Math.PI / 2;
    this.scene.add(m);
  }

  _addArc() {
    const pts = [];
    for (let a = 0; a <= Math.PI; a += 0.05) {
      pts.push(new THREE.Vector3(0, 3.0 * Math.cos(a) * 1.15, 3.0 * Math.sin(a) * 1.05));
    }
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({ color: 0x1e4080, transparent: true, opacity: 0.25 });
    this.scene.add(new THREE.Line(geo, mat));
  }

  /* ── Electrodes ──────────────────────────────────────────────────── */
  _buildElectrodes() {
    const baseGeo = new THREE.SphereGeometry(0.10, 14, 14);

    for (let i = 0; i < this.nChannels; i++) {
      // Golden spiral on hemisphere
      const phi   = Math.acos(1 - (i / this.nChannels) * 0.92);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;

      const rx = 2.92, ry = 2.92 * 1.15, rz = 2.92 * 1.05;
      const x  = rx * Math.sin(phi) * Math.cos(theta);
      const y  = ry * Math.cos(phi);
      const z  = rz * Math.sin(phi) * Math.sin(theta);

      // Core electrode — phong so emissive works
      const coreMat = new THREE.MeshPhongMaterial({
        color:    0x06b6d4,
        emissive: 0x000000,
        shininess: 60,
        transparent: true,
        opacity: 0.95,
      });
      const core = new THREE.Mesh(baseGeo, coreMat);
      core.position.set(x, y, z);
      this.scene.add(core);

      this.nodes.push({ core, x, y, z, phi, theta, power: 0 });
    }
  }

  /* ── Connection Mesh ─────────────────────────────────────────────── */
  _buildConnectionMesh() {
    // Pre-allocate for up to 200 connection lines (2 points × 3 coords each)
    this._maxEdges = 200;
    const positions = new Float32Array(this._maxEdges * 2 * 3);
    this._connGeo = new THREE.BufferGeometry();
    this._connGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this._connGeo.setDrawRange(0, 0);

    const connMat = new THREE.LineBasicMaterial({
      color:       0x06b6d4,
      transparent: true,
      opacity:     0.18,
      blending:    THREE.AdditiveBlending,
      depthWrite:  false,
    });
    this.connections = new THREE.LineSegments(this._connGeo, connMat);
    this.scene.add(this.connections);
  }

  /* ── Legend ──────────────────────────────────────────────────────── */
  _buildLegend() {
    // HTML legend overlay
    const legend = document.createElement('div');
    legend.style.cssText = `
      position: absolute; bottom: 10px; right: 12px;
      display: flex; flex-direction: column; align-items: flex-end;
      gap: 3px; pointer-events: none;
    `;
    const labels = ['MAX', '80%', '60%', '40%', '20%', 'MIN'];
    const colors = ['#ef4444','#f59e0b','#10b981','#06b6d4','#1a56a6','#0d1b4b'];
    labels.forEach((lbl, i) => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:6px;font-family:monospace;font-size:9px;color:#64748b;';
      row.innerHTML = `<span>${lbl}</span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${colors[i]};box-shadow:0 0 6px ${colors[i]}80"></span>`;
      legend.appendChild(row);
    });

    // badge
    const badge = document.createElement('div');
    badge.id = 'brain-mode-badge';
    badge.style.cssText = `
      position: absolute; top: 10px; left: 12px;
      font-family: monospace; font-size: 10px; color: #06b6d4;
      background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.2);
      padding: 3px 8px; border-radius: 4px; pointer-events: none;
    `;
    badge.textContent = '128 CH  ·  LIVE TOPOGRAPHY';

    this.container.style.position = 'relative';
    this.container.appendChild(legend);
    this.container.appendChild(badge);
  }

  /* ── Shockwave Ring ──────────────────────────────────────────────── */
  _spawnShockwave(cx, cy, cz) {
    const geo = new THREE.TorusGeometry(0.1, 0.04, 6, 32);
    const mat = new THREE.MeshBasicMaterial({
      color:      0xef4444,
      transparent: true,
      opacity:    0.9,
      blending:   THREE.AdditiveBlending,
      depthWrite: false,
    });
    const ring = new THREE.Mesh(geo, mat);
    // Orient ring to face outward from head center
    ring.position.set(cx, cy, cz);
    ring.lookAt(0, 0, 0);
    ring.rotateX(Math.PI / 2);
    this.scene.add(ring);
    this.shockwaves.push({ mesh: ring, age: 0 });
  }

  /* ── Spark Particle ──────────────────────────────────────────────── */
  _spawnSpark(fromNode, toNode) {
    if (this.sparks.length > 80) return; // cap
    const geo = new THREE.SphereGeometry(0.04, 4, 4);
    const mat = new THREE.MeshBasicMaterial({
      color:      0xffffff,
      transparent: true,
      opacity:    1.0,
      blending:   THREE.AdditiveBlending,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(fromNode.x, fromNode.y, fromNode.z);
    this.scene.add(mesh);
    this.sparks.push({
      mesh,
      from: new THREE.Vector3(fromNode.x, fromNode.y, fromNode.z),
      to:   new THREE.Vector3(toNode.x,   toNode.y,   toNode.z),
      t:    0, speed: 0.03 + Math.random() * 0.04,
    });
  }

  /* ─── Data Update ─────────────────────────────────────────────────── */
  updatePower(powerArray) {
    if (!powerArray || powerArray.length < 2) return;
    const len = Math.min(powerArray.length, this.nChannels);

    // Find instantaneous max
    let instantMax = 1e-6;
    for (let i = 0; i < len; i++) if (powerArray[i] > instantMax) instantMax = powerArray[i];

    // Update rolling EMA max (slow decay for a stable colour scale)
    this._emaMax = this._emaMax + this._EMA_ALPHA * (instantMax - this._emaMax);
    // Never let EMA drop below 10% of instant max (recovers quickly from silence)
    if (instantMax > this._emaMax * 2) this._emaMax = instantMax * 0.5;
    const scale = Math.max(this._emaMax, 1e-6) * 0.9;

    for (let i = 0; i < len; i++) {
      this.targetPowers[i] = Math.min(1.0, powerArray[i] / scale);
    }
  }

  triggerAlert(zone) {
    this.alertZone  = zone ? zone.toLowerCase() : '';
    this.alertDecay = 1.0;

    // Spawn shockwaves on matching nodes
    for (const node of this.nodes) {
      if (this._nodeInZone(node, this.alertZone)) {
        if (Math.random() < 0.12) this._spawnShockwave(node.x, node.y, node.z);
      }
    }
  }

  _nodeInZone(node, zone) {
    if (!zone) return true;
    const x = node.x, z = node.z;
    if (zone.includes('left')    && x < 0)         return true;
    if (zone.includes('right')   && x > 0)         return true;
    if (zone.includes('frontal') && z > 1.5)       return true;
    if (zone.includes('temporal') && Math.abs(x) > 1.8) return true;
    if (zone.includes('occipital') && z < -1.5)   return true;
    if (!zone.match(/left|right|frontal|temporal|occipital/)) return true;
    return false;
  }

  /* ── Color Map ───────────────────────────────────────────────────── */
  _heatColor(t) {
    // t in [0,1] → lerp through colorMap stops
    const stops = this.colorMap;
    const scaled = t * (stops.length - 1);
    const lo = Math.floor(scaled);
    const hi = Math.min(lo + 1, stops.length - 1);
    const f  = scaled - lo;
    return new THREE.Color().lerpColors(stops[lo], stops[hi], f);
  }

  /* ── Resize ──────────────────────────────────────────────────────── */
  _onResize() {
    const W = this.container.clientWidth;
    const H = this.container.clientHeight;
    if (!W || !H) return;
    this.camera.aspect = W / H;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(W, H);
  }

  /* ── Main Render Loop ────────────────────────────────────────────── */
  _animate() {
    requestAnimationFrame(() => this._animate());
    this.time += 0.016;

    if (this.controls) this.controls.update();
    if (this.alertDecay > 0) this.alertDecay -= 0.015;

    // Smooth all power values
    for (let i = 0; i < this.nChannels; i++) {
      this.currentPowers[i] += (this.targetPowers[i] - this.currentPowers[i]) * 0.12;
    }

    // --- Update electrodes ---
    // Find top-30 most active nodes for synapse rendering
    const sorted = this.nodes.map((n, i) => ({ n, i, p: this.currentPowers[i] }))
                              .sort((a, b) => b.p - a.p);
    const topActive = sorted.slice(0, 30);

    const connPositions = this._connGeo.attributes.position.array;
    let edgeCount = 0;

    // Occasionally spawn sparks between top nodes
    if (Math.random() < 0.18 && topActive.length >= 2) {
      const a = topActive[Math.floor(Math.random() * 10)].n;
      const b = topActive[Math.floor(Math.random() * 10)].n;
      if (a !== b && (a.power > 0.4 || b.power > 0.4)) this._spawnSpark(a, b);
    }

    for (let i = 0; i < this.nChannels; i++) {
      const node = this.nodes[i];
      const p    = this.currentPowers[i];
      node.power = p;

      // Determine color via heatmap
      let c = this._heatColor(p);

      // Alert zone highlight
      if (this.alertDecay > 0 && this._nodeInZone(node, this.alertZone)) {
        const pulse = 0.5 + 0.5 * Math.sin(this.time * 12);
        c.lerp(new THREE.Color(0xff2200), this.alertDecay * pulse * 0.9);
      }

      node.core.material.color.copy(c);
      node.core.material.emissive.copy(c).multiplyScalar(p * 0.4);
      node.core.material.opacity = 0.7 + p * 0.3;

      // Size grows slightly with power — max 1.8x
      const sc = 1.0 + p * 0.8;
      node.core.scale.setScalar(sc);

      // Animate inner brain colour (centroid of top-5)
      if (i < 5) {
        const avgP = sorted.slice(0, 5).reduce((s, v) => s + v.p, 0) / 5;
        const bc   = this._heatColor(avgP * 0.4);
        this.brainMesh.material.emissive.lerp(bc, 0.03);
      }
    }

    // --- Build synapse connections between top active pairs ---
    if (topActive.length >= 2) {
      for (let i = 0; i < Math.min(topActive.length - 1, 40) && edgeCount < this._maxEdges; i++) {
        const a = topActive[i].n;
        const b = topActive[i + 1].n;
        const minP = Math.min(a.power, b.power);
        if (minP < 0.25) continue;

        // only connect if reasonably close
        const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        if (dx*dx + dy*dy + dz*dz > 18) continue;

        const base = edgeCount * 6;
        connPositions[base]     = a.x; connPositions[base+1] = a.y; connPositions[base+2] = a.z;
        connPositions[base+3]   = b.x; connPositions[base+4] = b.y; connPositions[base+5] = b.z;
        edgeCount++;
      }
      this._connGeo.setDrawRange(0, edgeCount * 2);
      this._connGeo.attributes.position.needsUpdate = true;

      // Pulse connection opacity with activity
      const topP = topActive[0].p;
      this.connections.material.opacity = 0.08 + topP * 0.35;
    }

    // --- Update sparks ---
    for (let i = this.sparks.length - 1; i >= 0; i--) {
      const s = this.sparks[i];
      s.t += s.speed;
      if (s.t >= 1.0) {
        this.scene.remove(s.mesh);
        this.sparks.splice(i, 1);
        continue;
      }
      s.mesh.position.lerpVectors(s.from, s.to, s.t);
      s.mesh.material.opacity = Math.sin(s.t * Math.PI) * 0.95;
    }

    // --- Update shockwaves ---
    for (let i = this.shockwaves.length - 1; i >= 0; i--) {
      const sw = this.shockwaves[i];
      sw.age += 0.04;
      const sc = 1 + sw.age * 6;
      sw.mesh.scale.setScalar(sc);
      sw.mesh.material.opacity = Math.max(0, 0.85 * (1 - sw.age));
      if (sw.age >= 1) {
        this.scene.remove(sw.mesh);
        this.shockwaves.splice(i, 1);
      }
    }

    // Gentle brain mesh rotation (scene handles the rest via OrbitControls)
    this.brainMesh.rotation.y = this.time * 0.05;

    this.renderer.render(this.scene, this.camera);
  }
}
