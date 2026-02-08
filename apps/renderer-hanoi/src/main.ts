import * as THREE from "three";

type RecordingStep = {
  index: number;
  state_before?: any;
  state_after?: any;
  action?: { name: string; arguments: Record<string, any> };
  totals?: { moves: number; illegal_moves: number; tool_calls: number };
  legal?: boolean;
};

type Recording = {
  metadata?: Record<string, any>;
  summary?: Record<string, any>;
  steps: RecordingStep[];
};

class HanoiScene {
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
  private renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  private root = new THREE.Group();
  private diskMeshes = new Map<number, THREE.Mesh>();
  private pegMeshes: THREE.Mesh[] = [];
  private baseMesh: THREE.Mesh | null = null;

  private pegPositions: number[] = [];
  private diskHeight = 0.4;
  private diskGap = 0.05;
  private baseY = -0.2;

  constructor(container: HTMLElement) {
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color("#0b0d12");
    this.scene.add(this.root);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(5, 10, 5);
    this.scene.add(dir);

    this.camera.position.set(0, 6, 10);
    this.camera.lookAt(0, 0, 0);

    window.addEventListener("resize", () => this.resize(container));
    this.resize(container);
  }

  getCanvas(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  resize(container: HTMLElement) {
    const { clientWidth, clientHeight } = container;
    this.renderer.setSize(clientWidth, clientHeight);
    this.camera.aspect = clientWidth / clientHeight;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  setState(pegs: number[][], nDisks?: number) {
    const pegCount = Math.max(pegs.length, 1);
    const maxDisk = nDisks ?? Math.max(...pegs.flat(), 1);

    this.ensurePegs(pegCount);
    this.ensureDisks(maxDisk);

    for (let disk = 1; disk <= maxDisk; disk++) {
      const mesh = this.diskMeshes.get(disk);
      if (!mesh) continue;
      mesh.visible = false;
    }

    for (let pegIndex = 0; pegIndex < pegs.length; pegIndex++) {
      const peg = pegs[pegIndex];
      const x = this.pegPositions[pegIndex];
      for (let stackIndex = 0; stackIndex < peg.length; stackIndex++) {
        const diskSize = peg[stackIndex];
        const mesh = this.diskMeshes.get(diskSize);
        if (!mesh) continue;
        const y =
          this.baseY +
          this.diskHeight / 2 +
          stackIndex * (this.diskHeight + this.diskGap);
        mesh.position.set(x, y, 0);
        mesh.visible = true;
      }
    }

    this.render();
  }

  private ensurePegs(count: number) {
    if (this.pegMeshes.length === count) return;
    this.pegMeshes.forEach((mesh) => this.root.remove(mesh));
    this.pegMeshes = [];
    if (this.baseMesh) {
      this.root.remove(this.baseMesh);
      this.baseMesh = null;
    }

    const spacing = 3;
    const start = -((count - 1) * spacing) / 2;
    this.pegPositions = Array.from({ length: count }, (_, i) => start + i * spacing);

    const pegGeom = new THREE.CylinderGeometry(0.1, 0.1, 3, 16);
    const pegMat = new THREE.MeshStandardMaterial({ color: "#6b7280" });
    for (let i = 0; i < count; i++) {
      const mesh = new THREE.Mesh(pegGeom, pegMat);
      mesh.position.set(this.pegPositions[i], 1.2, 0);
      this.pegMeshes.push(mesh);
      this.root.add(mesh);
    }

    const baseGeom = new THREE.BoxGeometry(count * spacing + 2, 0.3, 2);
    const baseMat = new THREE.MeshStandardMaterial({ color: "#111827" });
    const base = new THREE.Mesh(baseGeom, baseMat);
    base.position.set(0, this.baseY, 0);
    this.baseMesh = base;
    this.root.add(base);

    this.updateCameraForPegCount(count);
  }

  private ensureDisks(maxDisk: number) {
    for (let disk = 1; disk <= maxDisk; disk++) {
      if (this.diskMeshes.has(disk)) continue;
      const ratio = maxDisk > 1 ? (disk - 1) / (maxDisk - 1) : 1;
      const radius = 0.5 + ratio * 1.2;
      const geom = new THREE.CylinderGeometry(radius, radius, this.diskHeight, 32);
      const color = new THREE.Color().setHSL(0.55 - ratio * 0.4, 0.7, 0.5);
      const mat = new THREE.MeshStandardMaterial({ color });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.visible = false;
      this.diskMeshes.set(disk, mesh);
      this.root.add(mesh);
    }
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  private updateCameraForPegCount(count: number) {
    const clampedCount = Math.max(1, count);
    const span = (clampedCount - 1) * 3;
    const z = Math.max(10, span * 1.15 + 6);
    const y = Math.max(6, 5.5 + clampedCount * 0.12);
    this.camera.position.set(0, y, z);
    this.camera.lookAt(0, 0.8, 0);
  }
}

class HanoiPlayer {
  private recording: Recording | null = null;
  private stepIndex = 0;
  private timer: number | null = null;
  private speed = 1;

  constructor(
    private scene: HanoiScene,
    private statusEl: HTMLElement,
    private actionEl: HTMLElement,
    private metaEl: HTMLElement
  ) {}

  loadRecording(recording: Recording) {
    this.recording = normalizeRecording(recording);
    this.stepIndex = 0;
    this.stop();
    this.update();
  }

  setSpeed(speed: number) {
    this.speed = speed;
    if (this.timer) {
      this.play();
    }
  }

  play() {
    this.stop();
    const interval = 800 / this.speed;
    this.timer = window.setInterval(() => {
      if (!this.recording) return;
      if (this.stepIndex >= this.recording.steps.length - 1) {
        this.stop();
        return;
      }
      this.stepIndex += 1;
      this.update();
    }, interval);
  }

  stop() {
    if (this.timer) {
      window.clearInterval(this.timer);
      this.timer = null;
    }
  }

  prev() {
    if (!this.recording) return;
    this.stepIndex = Math.max(0, this.stepIndex - 1);
    this.update();
  }

  next() {
    if (!this.recording) return;
    this.stepIndex = Math.min(this.recording.steps.length - 1, this.stepIndex + 1);
    this.update();
  }

  setStep(index: number) {
    if (!this.recording) return;
    this.stepIndex = Math.min(this.recording.steps.length - 1, Math.max(0, index));
    this.update();
  }

  getStepCount() {
    return this.recording?.steps.length ?? 0;
  }

  getStepIndex() {
    return this.stepIndex;
  }

  private extractState(step: RecordingStep): any {
    const raw = step.state_after ?? step.state_before ?? {};
    if (raw.state && raw.state.pegs) return raw.state;
    return raw;
  }

  update() {
    if (!this.recording) return;
    const step = this.recording.steps[this.stepIndex];
    const state = this.extractState(step);
    const nPegs = Number.isInteger(state.n_pegs) ? state.n_pegs : 3;
    const pegs = state.pegs ?? Array.from({ length: nPegs }, () => []);
    this.scene.setState(pegs, state.n_disks);

    const totals = step.totals ?? {};
    this.statusEl.textContent = `Step ${step.index} | moves=${totals.moves ?? 0} illegal=${
      totals.illegal_moves ?? 0
    } tool_calls=${totals.tool_calls ?? 0}`;
    this.actionEl.textContent = `Action: ${JSON.stringify(step.action ?? {})}`;
    this.metaEl.textContent = JSON.stringify(this.recording.metadata ?? {});
  }
}

function normalizeRecording(recording: Recording): Recording {
  const steps = recording.steps ?? [];
  if (!steps.length) return recording;
  if (steps[0]?.action == null) return recording;

  const meta = recording.metadata ?? {};
  let initialState = (meta as any).initial_state ?? steps[0]?.state_before;
  if (!initialState && typeof (meta as any).n_disks === "number") {
    const n = (meta as any).n_disks as number;
    const nPegs =
      typeof (meta as any).n_pegs === "number" ? (meta as any).n_pegs : 3;
    const startPegRaw =
      typeof (meta as any).start_peg === "number" ? (meta as any).start_peg : 0;
    const startPeg =
      startPegRaw >= 0 && startPegRaw < nPegs ? startPegRaw : 0;
    initialState = {
      n_disks: n,
      n_pegs: nPegs,
      pegs: Array.from({ length: nPegs }, (_, i) =>
        i === startPeg ? Array.from({ length: n }, (_, j) => n - j) : []
      ),
      disk_positions: Array.from({ length: n }, () => startPeg),
    };
  }
  if (!initialState) return recording;

  const initStep: RecordingStep = {
    index: 0,
    state_before: initialState,
    state_after: initialState,
    action: undefined,
    legal: true,
    totals: { moves: 0, illegal_moves: 0, tool_calls: 0 },
  };
  const normalizedSteps = [initStep, ...steps].map((step, idx) => ({
    ...step,
    index: idx,
  }));
  return { ...recording, steps: normalizedSteps };
}

const container = document.getElementById("canvasContainer") as HTMLElement;
const statusEl = document.getElementById("status") as HTMLElement;
const actionEl = document.getElementById("action") as HTMLElement;
const metaEl = document.getElementById("meta") as HTMLElement;

const scene = new HanoiScene(container);
const player = new HanoiPlayer(scene, statusEl, actionEl, metaEl);

const fileInput = document.getElementById("fileInput") as HTMLInputElement;
const loadSample = document.getElementById("loadSample") as HTMLButtonElement;
const urlInput = document.getElementById("urlInput") as HTMLInputElement;
const prevBtn = document.getElementById("prevBtn") as HTMLButtonElement;
const playBtn = document.getElementById("playBtn") as HTMLButtonElement;
const nextBtn = document.getElementById("nextBtn") as HTMLButtonElement;
const speedSlider = document.getElementById("speedSlider") as HTMLInputElement;
const stepSlider = document.getElementById("stepSlider") as HTMLInputElement;
const exportPng = document.getElementById("exportPng") as HTMLButtonElement;
const recordBtn = document.getElementById("recordBtn") as HTMLButtonElement;
const recordingStatus = document.getElementById("recordingStatus") as HTMLSpanElement;

let recorder: MediaRecorder | null = null;
let recordedChunks: Blob[] = [];

function updateSlider() {
  const count = player.getStepCount();
  stepSlider.max = Math.max(0, count - 1).toString();
  stepSlider.value = player.getStepIndex().toString();
}

function loadRecording(recording: Recording) {
  player.loadRecording(recording);
  updateSlider();
}

fileInput.addEventListener("change", async (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const text = await file.text();
  loadRecording(JSON.parse(text));
});

loadSample.addEventListener("click", async () => {
  const url = urlInput.value.trim();
  if (!url) return;
  const response = await fetch(url);
  const data = await response.json();
  loadRecording(data);
});

prevBtn.addEventListener("click", () => {
  player.prev();
  updateSlider();
});

nextBtn.addEventListener("click", () => {
  player.next();
  updateSlider();
});

playBtn.addEventListener("click", () => {
  if (playBtn.dataset.playing === "true") {
    player.stop();
    playBtn.textContent = "Play";
    playBtn.dataset.playing = "false";
  } else {
    player.play();
    playBtn.textContent = "Pause";
    playBtn.dataset.playing = "true";
  }
});

speedSlider.addEventListener("input", () => {
  player.setSpeed(parseFloat(speedSlider.value));
});

stepSlider.addEventListener("input", () => {
  player.setStep(parseInt(stepSlider.value, 10));
});

exportPng.addEventListener("click", () => {
  const link = document.createElement("a");
  link.download = "hanoi_frame.png";
  link.href = scene.getCanvas().toDataURL("image/png");
  link.click();
});

recordBtn.addEventListener("click", () => {
  if (!recorder) {
    const stream = scene.getCanvas().captureStream(30);
    recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
    recordedChunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) recordedChunks.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: "video/webm" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "hanoi_playback.webm";
      link.click();
      URL.revokeObjectURL(url);
      recordingStatus.textContent = "";
    };
    recorder.start();
    recordBtn.textContent = "Stop Recording";
    recordingStatus.textContent = "Recording...";
  } else {
    recorder.stop();
    recorder = null;
    recordBtn.textContent = "Record Video";
  }
});

// Expose a small API for programmatic control (vision benchmarks).
(window as any).hanoiRenderer = {
  setState: (pegs: number[][], nDisks?: number) => {
    scene.setState(pegs, nDisks);
  },
  setRecordingStep: (index: number) => {
    player.setStep(index);
  },
  exportPNG: () => scene.getCanvas().toDataURL("image/png"),
};
