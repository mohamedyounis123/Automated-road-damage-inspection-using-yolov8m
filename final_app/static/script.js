// RoadScan front-end logic — Image mode + Video upload & processing mode

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview-img");
const dzEmpty = document.getElementById("dropzone-empty");
const runBtn = document.getElementById("run-btn");
const scanline = document.getElementById("scanline");

const confSlider = document.getElementById("conf-slider");
const iouSlider = document.getElementById("iou-slider");
const confValue = document.getElementById("conf-value");
const iouValue = document.getElementById("iou-value");
const highAccToggle = document.getElementById("high-acc-toggle");

const statCount = document.getElementById("stat-count");
const statTime = document.getElementById("stat-time");
const detList = document.getElementById("detections-list");

let currentFile = null;

confSlider.addEventListener("input", () => (confValue.textContent = confSlider.value));
iouSlider.addEventListener("input", () => (iouValue.textContent = iouSlider.value));

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "#F2C14E";
});
dropzone.addEventListener("dragleave", () => (dropzone.style.borderColor = ""));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "";
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) return;
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.classList.remove("hidden");
    dzEmpty.classList.add("hidden");
    runBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

runBtn.addEventListener("click", async () => {
  if (!currentFile) return;
  runBtn.disabled = true;
  runBtn.textContent = "Scanning...";
  scanline.classList.add("active");

  const formData = new FormData();
  formData.append("image", currentFile);
  formData.append("conf", confSlider.value);
  formData.append("iou", iouSlider.value);
  formData.append("high_accuracy", highAccToggle.checked ? "true" : "false");

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      alert(data.error);
      return;
    }

    previewImg.src = data.image;
    statCount.textContent = data.count;
    statTime.textContent = data.inference_ms;

    detList.innerHTML = "";
    if (data.detections.length === 0) {
      detList.innerHTML = '<li class="det-empty">No damages detected — try lowering confidence threshold</li>';
    } else {
      data.detections.forEach((d) => {
        const li = document.createElement("li");
        li.innerHTML = `<span>${d.class}</span><span class="conf">${(d.confidence * 100).toFixed(1)}%</span>`;
        detList.appendChild(li);
      });
    }
  } catch (err) {
    alert("An error occurred while connecting to server: " + err.message);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run Detection ▸";
    scanline.classList.remove("active");
  }
});

document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".mode-view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.mode + "-mode").classList.add("active");
  });
});

// ============ Video Upload & Processing Mode ============

const videoDropzone = document.getElementById("video-dropzone");
const videoInput = document.getElementById("video-input");
const videoDzEmpty = document.getElementById("video-dz-empty");
const liveFeed = document.getElementById("live-feed");
const videoRunBtn = document.getElementById("video-run-btn");
const progressWrap = document.getElementById("video-progress-wrap");
const progressBar = document.getElementById("video-progress-bar");
const progressText = document.getElementById("video-progress-text");

let currentVideoFile = null;
let progressPollInterval = null;

videoDropzone.addEventListener("click", () => {
  if (!liveFeed.classList.contains("hidden")) return; // Ignore click during playback
  videoInput.click();
});

videoDropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  videoDropzone.style.borderColor = "#F2C14E";
});
videoDropzone.addEventListener("dragleave", () => (videoDropzone.style.borderColor = ""));
videoDropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  videoDropzone.style.borderColor = "";
  if (e.dataTransfer.files.length) handleVideoFile(e.dataTransfer.files[0]);
});

videoInput.addEventListener("change", (e) => {
  if (e.target.files.length) handleVideoFile(e.target.files[0]);
});

function handleVideoFile(file) {
  if (!file.type.startsWith("video/")) return;
  currentVideoFile = file;
  videoDzEmpty.innerHTML = `
    <span class="dz-icon">▶</span>
    <p>${file.name}</p>
    <p class="mono dz-hint">${(file.size / (1024 * 1024)).toFixed(1)} MB — Ready for processing</p>
  `;
  videoRunBtn.disabled = false;
}

videoRunBtn.addEventListener("click", async () => {
  // If previous processing finished and current file is empty, reset dropzone instead of sending
  if (!currentVideoFile) {
    liveFeed.src = "";
    liveFeed.classList.add("hidden");
    videoDzEmpty.classList.remove("hidden");
    videoDzEmpty.innerHTML = `
      <span class="dz-icon">⤒</span>
      <p>Drag and drop video file here or click to select</p>
      <p class="mono dz-hint">MP4 · AVI · MOV · MKV · WEBM — up to 300MB</p>
    `;
    progressWrap.classList.add("hidden");
    progressBar.style.width = "0%";
    videoRunBtn.textContent = "Process Video ▸";
    videoRunBtn.disabled = true;
    return;
  }
  videoRunBtn.disabled = true;
  videoRunBtn.textContent = "Uploading video...";

  const formData = new FormData();
  formData.append("video", currentVideoFile);
  formData.append("conf", confSlider.value);
  formData.append("iou", iouSlider.value);
  formData.append("high_accuracy", highAccToggle.checked ? "true" : "false");

  try {
    const res = await fetch("/upload_video", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      alert(data.error);
      videoRunBtn.disabled = false;
      videoRunBtn.textContent = "Process Video ▸";
      return;
    }

    // Show processed stream & hide upload dropzone
    videoDzEmpty.classList.add("hidden");
    liveFeed.classList.remove("hidden");
    liveFeed.src = `/video_feed/${data.session_id}?${Date.now()}`;

    progressWrap.classList.remove("hidden");
    progressText.textContent = `0 / ${data.total_frames} frames`;
    videoRunBtn.textContent = "Processing...";

    startProgressPolling(data.session_id, data.total_frames);
  } catch (err) {
    alert("An error occurred while uploading video: " + err.message);
    videoRunBtn.disabled = false;
    videoRunBtn.textContent = "Process Video ▸";
  }
});

function startProgressPolling(sessionId, totalFrames) {
  if (progressPollInterval) clearInterval(progressPollInterval);

  progressPollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/video_progress/${sessionId}`);
      const data = await res.json();
      if (data.error) {
        clearInterval(progressPollInterval);
        return;
      }

      const pct = totalFrames > 0
        ? Math.min(100, Math.round((data.processed_frames / totalFrames) * 100))
        : 0;
      progressBar.style.width = pct + "%";
      progressText.textContent = `${data.processed_frames} / ${totalFrames} frames — ${data.detections_total} detections`;

      if (data.done) {
        clearInterval(progressPollInterval);
        videoRunBtn.textContent = "Upload another video ▸";
        videoRunBtn.disabled = false;
        resetVideoUploadState();
      }
    } catch (err) {
      clearInterval(progressPollInterval);
    }
  }, 800);
}

function resetVideoUploadState() {
  // Allow uploading a new video after current processing finishes
  currentVideoFile = null;
  videoInput.value = "";
}

// Reset video upload UI when leaving mode (optional: stops current stream)
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.dataset.mode !== "live" && !liveFeed.classList.contains("hidden")) {
      liveFeed.src = "";
      liveFeed.classList.add("hidden");
      videoDzEmpty.classList.remove("hidden");
      videoDzEmpty.innerHTML = `
        <span class="dz-icon">⤒</span>
        <p>Drag and drop video file here or click to select</p>
        <p class="mono dz-hint">MP4 · AVI · MOV · MKV · WEBM — up to 300MB</p>
      `;
      progressWrap.classList.add("hidden");
      progressBar.style.width = "0%";
      videoRunBtn.textContent = "Process Video ▸";
      videoRunBtn.disabled = true;
      currentVideoFile = null;
      if (progressPollInterval) clearInterval(progressPollInterval);
    }
  });
});