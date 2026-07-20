(() => {
  "use strict";

  const els = {
    modeSpeak: document.getElementById("mode-speak"),
    modeType: document.getElementById("mode-type"),
    panelSpeak: document.getElementById("panel-speak"),
    panelType: document.getElementById("panel-type"),
    langSelect: document.getElementById("lang-select"),
    recordBtn: document.getElementById("record-btn"),
    recordLabel: document.getElementById("record-label"),
    recordWaveform: document.getElementById("record-waveform"),
    audioPreview: document.getElementById("audio-preview"),
    textInput: document.getElementById("text-input"),
    submitBtn: document.getElementById("submit-btn"),
    slipTimestamp: document.getElementById("slip-timestamp"),
    statusBanner: document.getElementById("status-banner"),
    statusText: document.getElementById("status-text"),
    resultsSection: document.getElementById("results-section"),
    resultTranscript: document.getElementById("result-transcript"),
    resultLangChip: document.getElementById("result-lang-chip"),
    resultEnglish: document.getElementById("result-english"),
    resultAnswer: document.getElementById("result-answer"),
    askAgainBtn: document.getElementById("ask-again-btn"),
    pulsePath: document.getElementById("pulse-path"),
  };

  let mode = "speak"; // "speak" | "type"
  let mediaRecorder = null;
  let recordedChunks = [];
  let recordedBlob = null;
  let isRecording = false;
  let languages = [];

  // -- language list -------------------------------------------------------

  async function loadLanguages() {
    try {
      const res = await fetch("/api/languages");
      languages = await res.json();
    } catch (err) {
      languages = []; // dropdown just keeps "Auto-detect" if this fails
    }
    for (const lang of languages) {
      const opt = document.createElement("option");
      opt.value = lang.code;
      opt.textContent = `${lang.name} · ${lang.native_name}`;
      els.langSelect.appendChild(opt);
    }
  }

  function langName(code) {
    const match = languages.find((l) => l.code === code);
    return match ? match.name : code;
  }

  // -- mode toggle -----------------------------------------------------------

  function setMode(next) {
    mode = next;
    const speaking = mode === "speak";
    els.modeSpeak.classList.toggle("is-active", speaking);
    els.modeType.classList.toggle("is-active", !speaking);
    els.modeSpeak.setAttribute("aria-selected", String(speaking));
    els.modeType.setAttribute("aria-selected", String(!speaking));
    els.panelSpeak.classList.toggle("hidden", !speaking);
    els.panelType.classList.toggle("hidden", speaking);
    updateSubmitState();
  }

  els.modeSpeak.addEventListener("click", () => setMode("speak"));
  els.modeType.addEventListener("click", () => setMode("type"));

  // -- recording ---------------------------------------------------------

  async function toggleRecording() {
    if (isRecording) {
      mediaRecorder.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunks.push(e.data);
      };
      mediaRecorder.onstop = () => {
        recordedBlob = new Blob(recordedChunks, { type: "audio/webm" });
        els.audioPreview.src = URL.createObjectURL(recordedBlob);
        els.audioPreview.classList.remove("hidden");
        stream.getTracks().forEach((t) => t.stop());
        setRecordingUI(false);
        updateSubmitState();
      };
      mediaRecorder.start();
      setRecordingUI(true);
    } catch (err) {
      els.recordLabel.textContent = "Microphone access denied";
    }
  }

  function setRecordingUI(active) {
    isRecording = active;
    els.recordBtn.classList.toggle("is-recording", active);
    els.recordBtn.setAttribute("aria-pressed", String(active));
    els.recordWaveform.classList.toggle("is-active", active);
    els.recordLabel.textContent = active ? "Tap to stop" : "Tap to speak again";
  }

  els.recordBtn.addEventListener("click", toggleRecording);

  // -- submit gating -------------------------------------------------------

  function updateSubmitState() {
    const ready = mode === "speak" ? !!recordedBlob : els.textInput.value.trim().length > 0;
    els.submitBtn.disabled = !ready;
  }

  els.textInput.addEventListener("input", updateSubmitState);

  // -- pulse rail energizing during requests ------------------------------

  let pulseInterval = null;

  function energizePulse(active) {
    if (pulseInterval) clearInterval(pulseInterval);
    if (!active) {
      els.pulsePath.setAttribute("d", "M0,20 L1200,20");
      return;
    }
    pulseInterval = setInterval(() => {
      let d = "M0,20 ";
      for (let x = 0; x <= 1200; x += 24) {
        const y = 20 + (Math.random() - 0.5) * 26;
        d += `L${x},${y.toFixed(1)} `;
      }
      els.pulsePath.setAttribute("d", d);
    }, 120);
  }

  // -- submit --------------------------------------------------------------

  async function submit() {
    els.submitBtn.disabled = true;
    els.statusBanner.classList.remove("hidden");
    els.statusText.textContent = mode === "speak"
      ? "Transcribing and translating…"
      : "Translating and thinking…";
    energizePulse(true);

    try {
      let response;
      const language = els.langSelect.value;

      if (mode === "speak") {
        const form = new FormData();
        form.append("audio", recordedBlob, "recording.webm");
        form.append("language", language);
        response = await fetch("/api/ask/audio", { method: "POST", body: form });
      } else {
        response = await fetch("/api/ask/text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: els.textInput.value.trim(), language }),
        });
      }

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Something went wrong." }));
        throw new Error(err.detail || "Something went wrong.");
      }

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      els.statusText.textContent = err.message || "Something went wrong. Please try again.";
      energizePulse(false);
      setTimeout(() => els.statusBanner.classList.add("hidden"), 3500);
      updateSubmitState();
      return;
    }

    energizePulse(false);
    els.statusBanner.classList.add("hidden");
    updateSubmitState();
  }

  els.submitBtn.addEventListener("click", submit);

  function renderResults(data) {
    els.resultTranscript.textContent = data.transcript;
    els.resultLangChip.textContent = data.detected_language_name || langName(data.detected_language);
    els.resultEnglish.textContent = data.english_text;
    els.resultAnswer.textContent = data.answer;

    els.resultsSection.classList.remove("hidden");
    els.resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  els.askAgainBtn.addEventListener("click", () => {
    recordedBlob = null;
    recordedChunks = [];
    els.audioPreview.classList.add("hidden");
    els.textInput.value = "";
    els.resultsSection.classList.add("hidden");
    updateSubmitState();
    document.getElementById("intake-slip").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  // -- init ------------------------------------------------------------------

  function initTimestamp() {
    const now = new Date();
    els.slipTimestamp.textContent = now.toLocaleString(undefined, {
      day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
    });
  }

  loadLanguages();
  initTimestamp();
  updateSubmitState();
})();
