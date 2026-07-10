import React from "react";
import ReactDOM from "react-dom/client";
import {
  Activity,
  Check,
  Circle,
  FolderOpen,
  Import,
  Pause,
  Play,
  Plus,
  Scissors,
  Settings,
  SlidersHorizontal,
  Square,
  Video,
} from "lucide-react";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import "./styles.css";

type ToolStatus = {
  name: string;
  path: string | null;
  available: boolean;
  version: string | null;
  source: string;
};

type DoctorReport = {
  ffmpeg: ToolStatus;
  ffprobe: ToolStatus;
  remediation: string[];
};

type VideoInfo = {
  path: string;
  codec?: string | null;
  width?: number | null;
  height?: number | null;
  fps?: number | null;
  duration_seconds?: number | null;
  frame_count?: number | null;
};

type PreviewFrame = {
  path: string;
  timestamp_seconds: number;
};

type CropSpace = "source" | "output";
type ImageFormat = "png" | "jpeg" | "webp";
type ColorSpace = "bgr" | "rgb" | "gray" | "hsv";
type ManifestFormat = "jsonl" | "csv";
type JobStatus = "idle" | "running" | "completed" | "failed" | "cancel_requested";
type CropDecision = "unset" | "crop" | "none";

type ExtractionReport = {
  input: string;
  output_dir: string;
  frames_written: number;
  manifests_written: string[];
};

type VideoJob = {
  input: string;
  output_dir: string;
  sampling: {
    every_n_frames: number;
    limit: number | null;
  };
  transforms: {
    crop: { x: number; y: number; width: number; height: number } | null;
    crop_space: CropSpace;
    resize: { width: number; height: number } | null;
    color_space: ColorSpace | null;
  };
  format: ImageFormat;
  prefix: string;
  start_number: number;
  manifests: ManifestFormat[];
  runtime: {
    ffmpeg_path: string | null;
    ffprobe_path: string | null;
    workers: number;
    overwrite: boolean;
  };
};

type FormState = {
  input: string;
  outputDir: string;
  everyNFrames: number;
  limit: number;
  cropSpace: CropSpace;
  cropX: number;
  cropY: number;
  cropW: number;
  cropH: number;
  resize: string;
  format: ImageFormat;
  prefix: string;
  startNumber: number;
  manifestJsonl: boolean;
  manifestCsv: boolean;
  colorSpace: ColorSpace | "";
  ffmpegPath: string;
  ffprobePath: string;
  workers: number;
  overwrite: boolean;
};

type QueueItem = {
  id: number;
  input: string;
  output: string;
  sampling: string;
  status: JobStatus;
  framesWritten: number | null;
  error?: string;
};

type CropRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type DragState = {
  mode: "draw" | "move" | "resize";
  handle?: "tl" | "tr" | "bl" | "br";
  start: { x: number; y: number };
  initial: CropRect;
};

const initialForm: FormState = {
  input: "",
  outputDir: "",
  everyNFrames: 5,
  limit: 0,
  cropSpace: "source",
  cropX: 0,
  cropY: 0,
  cropW: 0,
  cropH: 0,
  resize: "",
  format: "png",
  prefix: "frame",
  startNumber: 0,
  manifestJsonl: true,
  manifestCsv: true,
  colorSpace: "",
  ffmpegPath: "",
  ffprobePath: "",
  workers: 1,
  overwrite: false,
};

const steps = [
  { label: "Input", detail: "Choose video files", icon: Video },
  { label: "Crop Preview", detail: "Define or skip crop", icon: Scissors },
  { label: "Sampling", detail: "Confirm frame interval", icon: SlidersHorizontal },
  { label: "Output", detail: "Images and manifests", icon: FolderOpen },
  { label: "Run", detail: "Queue and execute", icon: Activity },
];

const statusText = {
  unset: "Needs decision",
  crop: "Crop selected",
  none: "No crop",
} satisfies Record<CropDecision, string>;

function App() {
  const [tab, setTab] = React.useState<"guided" | "advanced">("guided");
  const [form, setForm] = React.useState<FormState>(initialForm);
  const [cropDecision, setCropDecision] = React.useState<CropDecision>("unset");
  const [samplingConfirmed, setSamplingConfirmed] = React.useState(false);
  const [outputConfirmed, setOutputConfirmed] = React.useState(false);
  const [doctor, setDoctor] = React.useState<DoctorReport | null>(null);
  const [videoInfo, setVideoInfo] = React.useState<VideoInfo | null>(null);
  const [videoSrc, setVideoSrc] = React.useState("");
  const [videoError, setVideoError] = React.useState("");
  const [duration, setDuration] = React.useState(0);
  const [currentTime, setCurrentTime] = React.useState(0);
  const [frameInput, setFrameInput] = React.useState("0");
  const [timeInput, setTimeInput] = React.useState("0.000");
  const [playing, setPlaying] = React.useState(false);
  const [queue, setQueue] = React.useState<QueueItem[]>([]);
  const [activeRunId, setActiveRunId] = React.useState<number | null>(null);
  const [log, setLog] = React.useState<string[]>([
    "Ready. FFmpeg doctor runs automatically; load a video to start.",
  ]);

  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const stageRef = React.useRef<HTMLDivElement | null>(null);
  const drag = React.useRef<DragState | null>(null);
  const cancelledRun = React.useRef<number | null>(null);
  const doctorStarted = React.useRef(false);

  const job = React.useMemo(() => buildJob(form, cropDecision), [form, cropDecision]);
  const cropEnabled = cropDecision === "crop" && form.cropW > 0 && form.cropH > 0;
  const completion = React.useMemo(
    () => stepCompletion(form, videoInfo, cropDecision, samplingConfirmed, outputConfirmed, queue),
    [form, videoInfo, cropDecision, samplingConfirmed, outputConfirmed, queue],
  );
  const activeStep = completion.findIndex((done) => !done);
  const activeStepIndex = activeStep === -1 ? steps.length - 1 : activeStep;
  const canRun = completion.slice(0, 4).every(Boolean) && activeRunId === null;
  const fps = resolvedFps(videoInfo);
  const frameNumber = fps ? Math.round(currentTime * fps) : 0;

  React.useEffect(() => {
    if (!doctorStarted.current) {
      doctorStarted.current = true;
      if (hasNativeRuntime()) {
        void runDoctor(false);
      } else {
        pushLog("Frontend preview mode. Run the Tauri desktop app for doctor, dialogs, and extraction.");
      }
    }
  }, []);

  React.useEffect(() => {
    if (!form.input) {
      setVideoSrc("");
      setVideoInfo(null);
      setCurrentTime(0);
      setDuration(0);
      return;
    }
    setVideoError("");
    if (!hasNativeRuntime()) {
      setVideoSrc(convertFileSrc(form.input));
      return;
    }
    let cancelled = false;
    invoke("allow_asset_path_command", { path: form.input })
      .catch((error) => {
        pushLog(`Preview scope update failed: ${String(error)}`);
      })
      .finally(() => {
        if (!cancelled) {
          setVideoSrc(convertFileSrc(form.input));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [form.input]);

  React.useEffect(() => {
    setFrameInput(String(frameNumber));
    setTimeInput(currentTime.toFixed(3));
  }, [currentTime, frameNumber]);

  async function runDoctor(writeLogs = true) {
    if (!hasNativeRuntime()) {
      pushLog("Doctor requires the Tauri desktop runtime.");
      return;
    }
    try {
      const report = await invoke<DoctorReport>("doctor_command", {
        ffmpeg: form.ffmpegPath || null,
        ffprobe: form.ffprobePath || null,
      });
      setDoctor(report);
      if (writeLogs) {
        pushLog(`FFmpeg: ${report.ffmpeg.available ? "found" : "missing"} (${report.ffmpeg.source})`);
        pushLog(`FFprobe: ${report.ffprobe.available ? "found" : "missing"} (${report.ffprobe.source})`);
        report.remediation.forEach(pushLog);
      } else {
        pushLog(`Startup doctor: FFmpeg ${report.ffmpeg.available ? "ready" : "missing"} (${report.ffmpeg.source}).`);
      }
    } catch (error) {
      pushLog(`Doctor failed: ${String(error)}`);
    }
  }

  async function inspectSelectedVideo(path = form.input) {
    if (!hasNativeRuntime()) {
      pushLog("Inspect requires the Tauri desktop runtime.");
      return;
    }
    if (!path) {
      pushLog("Select an input video before inspecting.");
      return;
    }
    try {
      const info = await invoke<VideoInfo>("inspect_video_command", {
        path,
        ffprobe: form.ffprobePath || null,
      });
      setVideoInfo(info);
      pushLog(`Inspected ${shortPath(path)}: ${info.width ?? "?"}x${info.height ?? "?"}, ${formatFps(info.fps)} fps`);
    } catch (error) {
      setVideoInfo(null);
      pushLog(`Inspect failed: ${String(error)}`);
    }
  }

  async function extractPreviewFrame() {
    if (!hasNativeRuntime()) {
      pushLog("Preview frame extraction requires the Tauri desktop runtime.");
      return;
    }
    if (!form.input) {
      pushLog("Load a video before extracting a preview frame.");
      return;
    }
    try {
      const preview = await invoke<PreviewFrame>("preview_frame_command", {
        path: form.input,
        timestampSeconds: currentTime,
        ffmpeg: form.ffmpegPath || null,
      });
      pushLog(`Preview frame extracted at ${preview.timestamp_seconds.toFixed(3)}s: ${preview.path}`);
    } catch (error) {
      pushLog(`Preview frame extraction failed: ${String(error)}`);
    }
  }

  async function runExtraction() {
    if (!hasNativeRuntime()) {
      pushLog("Extraction requires the Tauri desktop runtime.");
      return;
    }
    if (!canRun) {
      pushLog("Complete input, crop/no-crop, sampling, and output before running.");
      return;
    }
    const runId = Date.now();
    cancelledRun.current = null;
    setActiveRunId(runId);
    const item: QueueItem = {
      id: runId,
      input: form.input,
      output: form.outputDir,
      sampling: `Every ${form.everyNFrames} frame${form.everyNFrames === 1 ? "" : "s"}`,
      status: "running",
      framesWritten: null,
    };
    setQueue((items) => [item, ...items]);
    pushLog(`Started extraction: ${shortPath(form.input)} -> ${form.outputDir}`);

    try {
      const report = await invoke<ExtractionReport>("extract_video_command", { job });
      if (cancelledRun.current === runId) {
        updateQueue(runId, { status: "cancel_requested", framesWritten: report.frames_written });
        pushLog(`Cancellation requested after FFmpeg completed ${report.frames_written} frame(s).`);
      } else {
        updateQueue(runId, { status: "completed", framesWritten: report.frames_written });
        pushLog(`Completed extraction: ${report.frames_written} frame(s), manifests: ${report.manifests_written.length}`);
      }
    } catch (error) {
      updateQueue(runId, { status: "failed", error: String(error), framesWritten: 0 });
      pushLog(`Extraction failed: ${String(error)}`);
    } finally {
      setActiveRunId(null);
    }
  }

  function requestCancel() {
    if (activeRunId === null) {
      return;
    }
    cancelledRun.current = activeRunId;
    updateQueue(activeRunId, { status: "cancel_requested" });
    pushLog("Cancellation requested. v1 marks the run and ignores stale completion; direct FFmpeg process termination is a follow-up hardening task.");
  }

  async function chooseInput() {
    if (!hasNativeRuntime()) {
      pushLog("File dialogs require the Tauri desktop runtime.");
      return;
    }
    try {
      const selected = await open({
        multiple: false,
        directory: false,
        filters: [{ name: "Video", extensions: ["mp4", "mov", "m4v", "mkv", "avi", "webm"] }],
      });
      if (typeof selected === "string") {
        selectInput(selected);
      }
    } catch (error) {
      pushLog(`Open video dialog failed: ${String(error)}`);
    }
  }

  async function chooseOutput() {
    if (!hasNativeRuntime()) {
      pushLog("Folder dialogs require the Tauri desktop runtime.");
      return;
    }
    try {
      const selected = await open({ multiple: false, directory: true });
      if (typeof selected === "string") {
        setFormValue("outputDir", selected);
      }
    } catch (error) {
      pushLog(`Open output dialog failed: ${String(error)}`);
    }
  }

  async function importLegacy() {
    if (!hasNativeRuntime()) {
      pushLog("Legacy import requires the Tauri desktop runtime.");
      return;
    }
    try {
      const selected = await open({
        multiple: false,
        directory: false,
        filters: [{ name: "YAML", extensions: ["yaml", "yml"] }],
      });
      if (typeof selected !== "string") {
        return;
      }
      const imported = await invoke<VideoJob>("import_legacy_config_command", {
        path: selected,
        output: null,
      });
      applyJobToForm(imported);
      pushLog(`Imported legacy config: ${selected}`);
    } catch (error) {
      pushLog(`Legacy import failed: ${String(error)}`);
    }
  }

  function selectInput(path: string) {
    setForm((current) => ({ ...current, input: path }));
    setCropDecision("unset");
    setSamplingConfirmed(false);
    setOutputConfirmed(false);
    void inspectSelectedVideo(path);
  }

  function updateQueue(id: number, patch: Partial<QueueItem>) {
    setQueue((items) => items.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  }

  function pushLog(line: string) {
    setLog((items) => [`${new Date().toLocaleTimeString()} ${line}`, ...items].slice(0, 80));
  }

  function setFormValue<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((current) => ({ ...current, [key]: value }));
    if (["cropX", "cropY", "cropW", "cropH", "cropSpace", "resize"].includes(String(key))) {
      if (key !== "resize") {
        setCropDecision("crop");
      }
    }
    if (["everyNFrames", "limit"].includes(String(key))) {
      setSamplingConfirmed(false);
    }
    if (["outputDir", "format", "manifestJsonl", "manifestCsv", "prefix", "startNumber"].includes(String(key))) {
      setOutputConfirmed(false);
    }
  }

  function setCrop(rect: CropRect, decision: CropDecision = "crop") {
    const bounds = cropBasisSize(form, videoInfo);
    const normalized = normalizeCrop(rect, bounds.width, bounds.height);
    setForm((current) => ({
      ...current,
      cropX: normalized.x,
      cropY: normalized.y,
      cropW: normalized.width,
      cropH: normalized.height,
    }));
    setCropDecision(decision);
  }

  function skipCrop() {
    setForm((current) => ({ ...current, cropX: 0, cropY: 0, cropW: 0, cropH: 0 }));
    setCropDecision("none");
  }

  function confirmCrop() {
    if (form.cropW > 0 && form.cropH > 0) {
      setCropDecision("crop");
    } else {
      pushLog("Draw a crop rectangle or choose No Crop.");
    }
  }

  function seekToTime(value: number) {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    const target = clamp(value, 0, video.duration || duration || 0);
    video.currentTime = target;
    setCurrentTime(target);
  }

  function seekToFrame(raw: string) {
    setFrameInput(raw);
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed) || !fps) {
      return;
    }
    seekToTime(parsed / fps);
  }

  function handleVideoMetadata() {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    setVideoError("");
    setDuration(Number.isFinite(video.duration) ? video.duration : videoInfo?.duration_seconds ?? 0);
    if (!videoInfo && video.videoWidth && video.videoHeight) {
      setVideoInfo({
        path: form.input,
        width: video.videoWidth,
        height: video.videoHeight,
        duration_seconds: Number.isFinite(video.duration) ? video.duration : null,
      });
    }
  }

  function togglePlayback() {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  }

  function pointerPoint(event: React.PointerEvent) {
    const stage = stageRef.current;
    if (!stage) {
      return null;
    }
    const stageRect = stage.getBoundingClientRect();
    const mediaRect = containedMediaRect(stageRect, videoInfo?.width, videoInfo?.height);
    const basis = cropBasisSize(form, videoInfo);
    const x = clamp(((event.clientX - mediaRect.left) / mediaRect.width) * basis.width, 0, basis.width);
    const y = clamp(((event.clientY - mediaRect.top) / mediaRect.height) * basis.height, 0, basis.height);
    return { x, y };
  }

  function handlePointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (!form.input || !videoInfo?.width || !videoInfo?.height) {
      return;
    }
    const point = pointerPoint(event);
    if (!point) {
      return;
    }
    const target = event.target as HTMLElement;
    const handle = target.dataset.handle as DragState["handle"] | undefined;
    const initial = currentCrop(form);
    if (handle) {
      drag.current = { mode: "resize", handle, start: point, initial };
    } else if (target.closest(".crop-box")) {
      drag.current = { mode: "move", start: point, initial };
    } else {
      drag.current = { mode: "draw", start: point, initial: { x: point.x, y: point.y, width: 0, height: 0 } };
      setCrop({ x: point.x, y: point.y, width: 1, height: 1 });
    }
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: React.PointerEvent<HTMLDivElement>) {
    if (!drag.current) {
      return;
    }
    const point = pointerPoint(event);
    if (!point) {
      return;
    }
    const bounds = cropBasisSize(form, videoInfo);
    const state = drag.current;
    if (state.mode === "draw") {
      setCrop(rectFromPoints(state.start, point));
      return;
    }
    if (state.mode === "move") {
      const dx = point.x - state.start.x;
      const dy = point.y - state.start.y;
      setCrop({
        ...state.initial,
        x: clamp(state.initial.x + dx, 0, bounds.width - state.initial.width),
        y: clamp(state.initial.y + dy, 0, bounds.height - state.initial.height),
      });
      return;
    }
    setCrop(resizeCrop(state.initial, point, state.handle ?? "br"));
  }

  function handlePointerUp(event: React.PointerEvent<HTMLDivElement>) {
    if (drag.current) {
      drag.current = null;
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function applyJobToForm(imported: VideoJob) {
    setForm((current) => ({
      ...current,
      input: imported.input,
      outputDir: imported.output_dir,
      everyNFrames: imported.sampling.every_n_frames,
      limit: imported.sampling.limit ?? 0,
      cropSpace: imported.transforms.crop_space,
      cropX: imported.transforms.crop?.x ?? 0,
      cropY: imported.transforms.crop?.y ?? 0,
      cropW: imported.transforms.crop?.width ?? 0,
      cropH: imported.transforms.crop?.height ?? 0,
      resize: imported.transforms.resize
        ? `${imported.transforms.resize.width}x${imported.transforms.resize.height}`
        : "",
      format: imported.format,
      prefix: imported.prefix,
      startNumber: imported.start_number,
      manifestJsonl: imported.manifests.includes("jsonl"),
      manifestCsv: imported.manifests.includes("csv"),
      colorSpace: imported.transforms.color_space ?? "",
      ffmpegPath: imported.runtime.ffmpeg_path ?? "",
      ffprobePath: imported.runtime.ffprobe_path ?? "",
      workers: imported.runtime.workers,
      overwrite: imported.runtime.overwrite,
    }));
    setCropDecision(imported.transforms.crop ? "crop" : "unset");
    setSamplingConfirmed(false);
    setOutputConfirmed(false);
    if (imported.input) {
      void inspectSelectedVideo(imported.input);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Video size={18} /></span>
          <strong>vid2dataset</strong>
        </div>
        <nav>
          <button onClick={chooseInput}>Open Video</button>
          <button onClick={chooseOutput}>Output Folder</button>
          <button onClick={importLegacy}><Import size={14} /> Import Legacy</button>
        </nav>
        <button className="status" onClick={() => runDoctor(true)}>
          <Circle size={9} fill={doctor?.ffmpeg.available ? "currentColor" : "transparent"} />
          {doctor?.ffmpeg.available ? `FFmpeg ${doctor.ffmpeg.source}` : "Doctor"}
        </button>
      </header>

      <section className="workspace">
        <aside className="steps">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const done = completion[index];
            return (
              <button className={index === activeStepIndex ? "step active" : "step"} key={step.label}>
                <span className="step-index">{index + 1}</span>
                <Icon size={18} />
                <span>
                  <strong>{step.label}</strong>
                  <small>{step.detail}</small>
                </span>
                {done && <Check size={16} className="done" />}
              </button>
            );
          })}
          <button className="side-action" onClick={() => setTab("advanced")}><Settings size={17} /> Settings</button>
        </aside>

        <section className="preview-panel">
          <div className="panel-title">
            <strong>Crop Preview</strong>
            <span>{form.cropSpace === "source" ? "Source pixels" : "Output pixels"} · {videoSizeLabel(videoInfo)}</span>
          </div>
          <div
            ref={stageRef}
            className={`video-stage${form.input ? "" : " empty"}`}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
          >
            {form.input ? (
              <>
                <video
                  ref={videoRef}
                  src={videoSrc}
                  preload="metadata"
                  onLoadedMetadata={handleVideoMetadata}
                  onLoadedData={() => setVideoError("")}
                  onCanPlay={() => setVideoError("")}
                  onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                  onError={() => setVideoError("Video preview failed to load. Try Inspect or verify the file path.")}
                />
                <div className="frame-label left">{videoError || videoSizeLabel(videoInfo)}</div>
                <div className="frame-label right">{statusText[cropDecision]}</div>
                {cropEnabled && (
                  <div className="crop-box" style={cropBoxStyle(form, videoInfo)}>
                    <i className="handle tl" data-handle="tl" />
                    <i className="handle tr" data-handle="tr" />
                    <i className="handle bl" data-handle="bl" />
                    <i className="handle br" data-handle="br" />
                    <span>{form.cropW}x{form.cropH}</span>
                  </div>
                )}
              </>
            ) : (
              <strong>Load video to start</strong>
            )}
          </div>
          <div className="scrubber">
            <button onClick={togglePlayback} disabled={!form.input}>
              {playing ? <Pause size={15} /> : <Play size={15} />}
            </button>
            <input
              type="range"
              min={0}
              max={duration || videoInfo?.duration_seconds || 0}
              step={fps ? 1 / fps : 0.01}
              value={currentTime}
              onChange={(event) => seekToTime(Number.parseFloat(event.target.value))}
              disabled={!form.input}
            />
            <span>{formatDuration(currentTime)} / {formatDuration(duration || videoInfo?.duration_seconds)}</span>
          </div>
          <div className="seek-grid">
            <label>Time (s)<input value={timeInput} onChange={(event) => setTimeInput(event.target.value)} onBlur={() => seekToTime(Number.parseFloat(timeInput) || 0)} /></label>
            <label>Frame<input value={frameInput} onChange={(event) => seekToFrame(event.target.value)} disabled={!fps} /></label>
            <button onClick={extractPreviewFrame} disabled={!form.input}>Extract Preview Frame</button>
          </div>
          <div className="metadata">
            <span>Input: {form.input ? shortPath(form.input) : "No video selected"}</span>
            <span>Codec: {videoInfo?.codec ?? "n/a"}</span>
            <span>FPS: {formatFps(videoInfo?.fps)}</span>
            <span>Duration: {formatDuration(videoInfo?.duration_seconds)}</span>
            <span>Frames: {videoInfo?.frame_count ?? "n/a"}</span>
          </div>
          <div className="transport">
            <button onClick={() => inspectSelectedVideo()}><Play size={15} /> Inspect</button>
            <button onClick={confirmCrop} disabled={!form.input}>Use Crop</button>
            <button onClick={skipCrop} disabled={!form.input}>No Crop</button>
            <span>{form.cropSpace === "output" ? "Output crop uses resize dimensions when resize is set." : "Crop coordinates are source-frame pixels."}</span>
          </div>
        </section>

        <aside className="inspector">
          <div className="tabs">
            <button className={tab === "guided" ? "selected" : ""} onClick={() => setTab("guided")}>Guided</button>
            <button className={tab === "advanced" ? "selected" : ""} onClick={() => setTab("advanced")}>Advanced</button>
          </div>
          {tab === "guided" ? (
            <GuidedPanel
              form={form}
              setFormValue={setFormValue}
              chooseInput={chooseInput}
              chooseOutput={chooseOutput}
              doctor={doctor}
              cropDecision={cropDecision}
              samplingConfirmed={samplingConfirmed}
              outputConfirmed={outputConfirmed}
              onConfirmSampling={() => setSamplingConfirmed(true)}
              onConfirmOutput={() => setOutputConfirmed(true)}
            />
          ) : (
            <AdvancedPanel form={form} setFormValue={setFormValue} />
          )}
        </aside>
      </section>

      <section className="run-area">
        <div className="queue-head">
          <strong>Run Queue</strong>
          <div>
            <button onClick={runExtraction} disabled={!canRun}><Plus size={15} /> Run Job</button>
            <button onClick={requestCancel} disabled={activeRunId === null}><Square size={15} /> Cancel</button>
            <button onClick={() => runDoctor(true)}>Run Doctor</button>
          </div>
        </div>
        <div className="queue-table">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Input</th>
                <th>Output</th>
                <th>Sampling</th>
                <th>Frames</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {queue.length === 0 ? (
                <tr>
                  <td colSpan={6}>No jobs run yet.</td>
                </tr>
              ) : queue.map((item, index) => (
                <tr key={item.id}>
                  <td>{queue.length - index}</td>
                  <td>{shortPath(item.input)}</td>
                  <td>{item.output}</td>
                  <td>{item.sampling}</td>
                  <td>{item.framesWritten ?? <span className="progress"><i /></span>}</td>
                  <td className={statusClass(item.status)}>{statusLabel(item)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="logs">
          <strong>Logs</strong>
          {log.map((line, index) => <code key={`${line}-${index}`}>{line}</code>)}
        </div>
      </section>
    </main>
  );
}

function GuidedPanel({
  form,
  setFormValue,
  chooseInput,
  chooseOutput,
  doctor,
  cropDecision,
  samplingConfirmed,
  outputConfirmed,
  onConfirmSampling,
  onConfirmOutput,
}: {
  form: FormState;
  setFormValue: <K extends keyof FormState>(key: K, value: FormState[K]) => void;
  chooseInput: () => void;
  chooseOutput: () => void;
  doctor: DoctorReport | null;
  cropDecision: CropDecision;
  samplingConfirmed: boolean;
  outputConfirmed: boolean;
  onConfirmSampling: () => void;
  onConfirmOutput: () => void;
}) {
  return (
    <div className="form">
      <label>
        Input video
        <div className="input-row">
          <input value={form.input} onChange={(event) => setFormValue("input", event.target.value)} placeholder="/path/to/video.mp4" />
          <button onClick={chooseInput}>Browse</button>
        </div>
      </label>
      <label>
        Output folder
        <div className="input-row">
          <input value={form.outputDir} onChange={(event) => setFormValue("outputDir", event.target.value)} placeholder="/path/to/dataset" />
          <button onClick={chooseOutput}>Browse</button>
        </div>
      </label>
      <div className="decision-row">
        <span>Crop</span>
        <strong className={cropDecision === "unset" ? "missing" : "ok"}>{statusText[cropDecision]}</strong>
      </div>
      <label>
        Every N frames
        <input type="number" min={1} value={form.everyNFrames} onChange={(event) => setFormValue("everyNFrames", numberValue(event, 1))} />
      </label>
      <button className="confirm" onClick={onConfirmSampling}>
        {samplingConfirmed ? <Check size={15} /> : null} Confirm Sampling
      </button>
      <label>
        Crop coordinate space
        <select value={form.cropSpace} onChange={(event) => setFormValue("cropSpace", event.target.value as CropSpace)}>
          <option value="source">Source (input video)</option>
          <option value="output">Output (after resize)</option>
        </select>
      </label>
      <div className="quad">
        <label>X<input type="number" min={0} value={form.cropX} onChange={(event) => setFormValue("cropX", numberValue(event, 0))} /></label>
        <label>Y<input type="number" min={0} value={form.cropY} onChange={(event) => setFormValue("cropY", numberValue(event, 0))} /></label>
        <label>W<input type="number" min={0} value={form.cropW} onChange={(event) => setFormValue("cropW", numberValue(event, 0))} /></label>
        <label>H<input type="number" min={0} value={form.cropH} onChange={(event) => setFormValue("cropH", numberValue(event, 0))} /></label>
      </div>
      <label>
        Resize output
        <input value={form.resize} onChange={(event) => setFormValue("resize", event.target.value)} placeholder="640x640 or blank" />
      </label>
      <label>
        Image format
        <select value={form.format} onChange={(event) => setFormValue("format", event.target.value as ImageFormat)}>
          <option value="png">PNG (.png)</option>
          <option value="jpeg">JPEG (.jpg)</option>
          <option value="webp">WebP (.webp)</option>
        </select>
      </label>
      <fieldset>
        <legend>Manifests</legend>
        <label><input type="checkbox" checked={form.manifestJsonl} onChange={(event) => setFormValue("manifestJsonl", event.target.checked)} /> JSONL</label>
        <label><input type="checkbox" checked={form.manifestCsv} onChange={(event) => setFormValue("manifestCsv", event.target.checked)} /> CSV</label>
      </fieldset>
      <button className="confirm" onClick={onConfirmOutput} disabled={!form.outputDir || (!form.manifestCsv && !form.manifestJsonl)}>
        {outputConfirmed ? <Check size={15} /> : null} Confirm Output
      </button>
      <div className="tool-status">
        <span>FFmpeg</span>
        <strong className={doctor?.ffmpeg.available ? "ok" : "missing"}>
          {doctor ? (doctor.ffmpeg.available ? `${doctor.ffmpeg.source}` : "missing") : "checking"}
        </strong>
      </div>
    </div>
  );
}

function AdvancedPanel({
  form,
  setFormValue,
}: {
  form: FormState;
  setFormValue: <K extends keyof FormState>(key: K, value: FormState[K]) => void;
}) {
  return (
    <div className="form">
      <label>FFmpeg path<input value={form.ffmpegPath} onChange={(event) => setFormValue("ffmpegPath", event.target.value)} placeholder="Bundled, env, system, or explicit path" /></label>
      <label>FFprobe path<input value={form.ffprobePath} onChange={(event) => setFormValue("ffprobePath", event.target.value)} placeholder="Bundled, env, system, or explicit path" /></label>
      <label>Worker count<input type="number" min={1} value={form.workers} onChange={(event) => setFormValue("workers", numberValue(event, 1))} /></label>
      <label>Filename prefix<input value={form.prefix} onChange={(event) => setFormValue("prefix", event.target.value)} /></label>
      <label>Start number<input type="number" min={0} value={form.startNumber} onChange={(event) => setFormValue("startNumber", numberValue(event, 0))} /></label>
      <label>Limit frames<input type="number" min={0} value={form.limit} onChange={(event) => setFormValue("limit", numberValue(event, 0))} /></label>
      <label>Color space<select value={form.colorSpace} onChange={(event) => setFormValue("colorSpace", event.target.value as ColorSpace | "")}><option value="">Original</option><option value="bgr">BGR</option><option value="rgb">RGB</option><option value="gray">GRAY</option><option value="hsv">HSV</option></select></label>
      <label className="check"><input type="checkbox" checked={form.overwrite} onChange={(event) => setFormValue("overwrite", event.target.checked)} /> Overwrite existing frames</label>
    </div>
  );
}

function buildJob(form: FormState, cropDecision: CropDecision): VideoJob {
  return {
    input: form.input,
    output_dir: form.outputDir,
    sampling: {
      every_n_frames: Math.max(1, form.everyNFrames),
      limit: form.limit > 0 ? form.limit : null,
    },
    transforms: {
      crop: cropDecision === "crop" && form.cropW > 0 && form.cropH > 0
        ? { x: form.cropX, y: form.cropY, width: form.cropW, height: form.cropH }
        : null,
      crop_space: form.cropSpace,
      resize: parseResize(form.resize),
      color_space: form.colorSpace || null,
    },
    format: form.format,
    prefix: form.prefix.trim() || "frame",
    start_number: form.startNumber,
    manifests: [
      ...(form.manifestJsonl ? ["jsonl" as const] : []),
      ...(form.manifestCsv ? ["csv" as const] : []),
    ],
    runtime: {
      ffmpeg_path: form.ffmpegPath || null,
      ffprobe_path: form.ffprobePath || null,
      workers: Math.max(1, form.workers),
      overwrite: form.overwrite,
    },
  };
}

function parseResize(raw: string): { width: number; height: number } | null {
  const normalized = raw.trim().toLowerCase().replace("x", ",");
  if (!normalized) {
    return null;
  }
  const [width, height] = normalized.split(",").map((value) => Number.parseInt(value.trim(), 10));
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }
  return { width, height };
}

function numberValue(event: React.ChangeEvent<HTMLInputElement>, fallback: number) {
  const value = Number.parseInt(event.target.value, 10);
  return Number.isFinite(value) ? value : fallback;
}

function stepCompletion(
  form: FormState,
  videoInfo: VideoInfo | null,
  cropDecision: CropDecision,
  samplingConfirmed: boolean,
  outputConfirmed: boolean,
  queue: QueueItem[],
) {
  return [
    Boolean(form.input && videoInfo),
    cropDecision !== "unset",
    samplingConfirmed,
    Boolean(outputConfirmed && form.outputDir && (form.manifestCsv || form.manifestJsonl)),
    queue.some((item) => item.status === "completed"),
  ];
}

function cropBoxStyle(form: FormState, videoInfo: VideoInfo | null): React.CSSProperties {
  const basis = cropBasisSize(form, videoInfo);
  const left = clamp((form.cropX / basis.width) * 100, 0, 100);
  const top = clamp((form.cropY / basis.height) * 100, 0, 100);
  const boxWidth = clamp((form.cropW / basis.width) * 100, 0, 100 - left);
  const boxHeight = clamp((form.cropH / basis.height) * 100, 0, 100 - top);
  return {
    left: `${left}%`,
    top: `${top}%`,
    width: `${boxWidth}%`,
    height: `${boxHeight}%`,
  };
}

function cropBasisSize(form: FormState, info: VideoInfo | null) {
  const resize = form.cropSpace === "output" ? parseResize(form.resize) : null;
  return {
    width: resize?.width || info?.width || 1920,
    height: resize?.height || info?.height || 1080,
  };
}

function currentCrop(form: FormState): CropRect {
  return {
    x: form.cropX,
    y: form.cropY,
    width: form.cropW,
    height: form.cropH,
  };
}

function normalizeCrop(rect: CropRect, maxWidth: number, maxHeight: number): CropRect {
  const normalized = rect.width < 0
    ? { ...rect, x: rect.x + rect.width, width: Math.abs(rect.width) }
    : { ...rect };
  if (normalized.height < 0) {
    normalized.y += normalized.height;
    normalized.height = Math.abs(normalized.height);
  }
  const x = Math.round(clamp(normalized.x, 0, maxWidth));
  const y = Math.round(clamp(normalized.y, 0, maxHeight));
  const width = Math.round(clamp(normalized.width, 0, maxWidth - x));
  const height = Math.round(clamp(normalized.height, 0, maxHeight - y));
  return { x, y, width, height };
}

function rectFromPoints(start: { x: number; y: number }, end: { x: number; y: number }): CropRect {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  };
}

function resizeCrop(initial: CropRect, point: { x: number; y: number }, handle: "tl" | "tr" | "bl" | "br"): CropRect {
  const right = initial.x + initial.width;
  const bottom = initial.y + initial.height;
  if (handle === "tl") {
    return { x: point.x, y: point.y, width: right - point.x, height: bottom - point.y };
  }
  if (handle === "tr") {
    return { x: initial.x, y: point.y, width: point.x - initial.x, height: bottom - point.y };
  }
  if (handle === "bl") {
    return { x: point.x, y: initial.y, width: right - point.x, height: point.y - initial.y };
  }
  return { x: initial.x, y: initial.y, width: point.x - initial.x, height: point.y - initial.y };
}

function containedMediaRect(stage: DOMRect, width?: number | null, height?: number | null) {
  if (!width || !height) {
    return stage;
  }
  const mediaRatio = width / height;
  const stageRatio = stage.width / stage.height;
  if (stageRatio > mediaRatio) {
    const mediaWidth = stage.height * mediaRatio;
    const left = stage.left + (stage.width - mediaWidth) / 2;
    return new DOMRect(left, stage.top, mediaWidth, stage.height);
  }
  const mediaHeight = stage.width / mediaRatio;
  const top = stage.top + (stage.height - mediaHeight) / 2;
  return new DOMRect(stage.left, top, stage.width, mediaHeight);
}

function resolvedFps(info: VideoInfo | null) {
  if (typeof info?.fps === "number" && info.fps > 0) {
    return info.fps;
  }
  if (info?.frame_count && info.duration_seconds && info.duration_seconds > 0) {
    return info.frame_count / info.duration_seconds;
  }
  return 0;
}

function videoSizeLabel(info: VideoInfo | null) {
  if (!info?.width || !info?.height) {
    return "No metadata";
  }
  return `${info.width}x${info.height}`;
}

function formatFps(value?: number | null) {
  return typeof value === "number" ? value.toFixed(2) : "n/a";
}

function formatDuration(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(3)}s` : "n/a";
}

function shortPath(path: string) {
  return path.split(/[\\/]/).filter(Boolean).pop() || path;
}

function statusClass(status: JobStatus) {
  if (status === "completed") return "running";
  if (status === "failed") return "missing";
  return "";
}

function statusLabel(item: QueueItem) {
  if (item.status === "failed") {
    return item.error || "Failed";
  }
  if (item.status === "cancel_requested") {
    return "Cancel requested";
  }
  if (item.status === "completed") {
    return "Completed";
  }
  if (item.status === "running") {
    return "Running";
  }
  return "Idle";
}

function clamp(value: number, min: number, max: number) {
  if (max < min) {
    return min;
  }
  return Math.min(max, Math.max(min, value));
}

function hasNativeRuntime() {
  return typeof window !== "undefined"
    && Boolean((window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__);
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
