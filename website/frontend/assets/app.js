const API_BASE = "http://127.0.0.1:8000";
(() => {
  "use strict";

  const els = {
    modeSpeak: document.getElementById("mode-speak"),
    modeType: document.getElementById("mode-type"),
    panelSpeak: document.getElementById("panel-speak"),
    panelType: document.getElementById("panel-type"),
    langSelect: document.getElementById("lang-select"),
    personaSelect: document.getElementById("persona-select"),
    recordBtn: document.getElementById("record-btn"),
    recordLabel: document.getElementById("record-label"),
    recordWaveform: document.getElementById("record-waveform"),
    audioPreview: document.getElementById("audio-preview"),
    textInput: document.getElementById("text-input"),
    submitBtn: document.getElementById("submit-btn"),
    slipTimestamp: document.getElementById("slip-timestamp"),
    memoryNote: document.getElementById("memory-note"),
    statusBanner: document.getElementById("status-banner"),
    statusText: document.getElementById("status-text"),
    resultsSection: document.getElementById("results-section"),
    emergencyBanner: document.getElementById("emergency-banner"),
    emergencyText: document.getElementById("emergency-text"),
    resultTranscript: document.getElementById("result-transcript"),
    resultLangChip: document.getElementById("result-lang-chip"),
    resultEnglish: document.getElementById("result-english"),
    triageTicket: document.getElementById("triage-ticket"),
    triageBadge: document.getElementById("triage-badge"),
    confidenceNote: document.getElementById("confidence-note"),
    conditionsRow: document.getElementById("conditions-row"),
    conditionsList: document.getElementById("conditions-list"),
    redflagsRow: document.getElementById("redflags-row"),
    redflagsList: document.getElementById("redflags-list"),
    functionNote: document.getElementById("function-note"),
    resultAnswer: document.getElementById("result-answer"),
    resultStamp: document.getElementById("result-stamp"),
    noteSources: document.getElementById("note-sources"),
    sourcesList: document.getElementById("sources-list"),
    listenBtn: document.getElementById("listen-btn"),
    answerAudio: document.getElementById("answer-audio"),
    askAgainBtn: document.getElementById("ask-again-btn"),
    newConvoBtn: document.getElementById("new-convo-btn"),
    pulsePath: document.getElementById("pulse-path"),
  };

  let mode = "speak"; // "speak" | "type"
  let mediaRecorder = null;
  let recordedChunks = [];
  let recordedBlob = null;
  let isRecording = false;
  let languages = [];
  let sessionId = null;          // set once the first response comes back
  let lastAnswerLanguage = "en"; // used by the "Listen" button
  let userLat = null;
  let userLng = null;

  // -- language list -------------------------------------------------------

  async function loadLanguages() {
    try {
      const res = await fetch(`${API_BASE}/api/languages`);
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
      const modePersona = els.personaSelect.value;

      if (mode === "speak") {
        const form = new FormData();
        form.append("audio", recordedBlob, "recording.webm");
        form.append("language", language);
        form.append("mode", modePersona);
        if (sessionId) form.append("session_id", sessionId);
        if (userLat !== null) form.append("lat", userLat);
        if (userLng !== null) form.append("lng", userLng);
        response = await fetch(`${API_BASE}/api/ask/audio`, { method: "POST", body: form });
      } else {
        response = await fetch(`${API_BASE}/api/ask/text`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: els.textInput.value.trim(),
            language,
            session_id: sessionId,
            mode: modePersona,
            lat: userLat,
            lng: userLng,
          }),
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

  const TRIAGE_LABELS = {
    self_care: "Self-care",
    routine: "Routine",
    urgent: "Urgent",
    emergency: "Emergency",
    unknown: "Assessment unavailable",
  };

  function renderResults(data) {
    sessionId = data.session_id;
    lastAnswerLanguage = data.detected_language || "en";

    els.resultTranscript.textContent = data.transcript;
    els.resultLangChip.textContent = data.detected_language_name || langName(data.detected_language);
    els.resultEnglish.textContent = data.english_text;
    els.resultAnswer.textContent = data.answer;

    // Emergency banner
    els.emergencyBanner.classList.toggle("hidden", !data.is_emergency);
    if (data.is_emergency && data.function_note) {
      els.emergencyText.innerHTML = data.function_note;
    }

    // Triage / assessment ticket (hidden entirely if the fallback path was used,
    // since there's no structured triage in that case)
    if (data.used_fallback) {
      els.triageTicket.classList.add("hidden");
    } else {
      els.triageTicket.classList.remove("hidden");
      const label = TRIAGE_LABELS[data.triage] || data.triage;
      els.triageBadge.textContent = label;
      els.triageBadge.className = `triage-badge triage-${data.triage}`;
      els.confidenceNote.textContent = data.confidence
        ? `${Math.round(data.confidence * 100)}% confidence`
        : "";

      if (data.possible_conditions && data.possible_conditions.length > 0) {
        els.conditionsRow.classList.remove("hidden");
        els.conditionsList.textContent = data.possible_conditions.join(", ");
      } else {
        els.conditionsRow.classList.add("hidden");
      }

      if (data.red_flags && data.red_flags.length > 0) {
        els.redflagsRow.classList.remove("hidden");
        els.redflagsList.textContent = data.red_flags.join(", ");
      } else {
        els.redflagsRow.classList.add("hidden");
      }

      if (data.function_note && !data.is_emergency) {
        els.functionNote.classList.remove("hidden");
        els.functionNote.innerHTML = data.function_note;
      } else {
        els.functionNote.classList.add("hidden");
      }
    }

    if (data.is_grounded && data.sources && data.sources.length > 0) {
      els.resultStamp.innerHTML = "WHO SOURCED<br><span>see link below</span>";
      els.resultStamp.classList.add("stamp-grounded");
      els.noteSources.classList.remove("hidden");
      els.sourcesList.innerHTML = data.sources
        .map((url) => `<a href="${url}" target="_blank" rel="noopener">${url}</a>`)
        .join(" ");
    } else {
      els.resultStamp.innerHTML = "DRAFT<br><span>pending clinical source</span>";
      els.resultStamp.classList.remove("stamp-grounded");
      els.noteSources.classList.add("hidden");
      els.sourcesList.innerHTML = "";
    }

    els.answerAudio.classList.add("hidden");
    els.answerAudio.src = "";
    els.memoryNote.classList.remove("hidden");

    els.resultsSection.classList.remove("hidden");
    els.resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // -- listen to the answer (TTS) -------------------------------------------

  async function listenToAnswer() {
    els.listenBtn.disabled = true;
    els.listenBtn.textContent = "Generating audio…";
    try {
      const response = await fetch(`${API_BASE}/api/speak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: els.resultAnswer.textContent, language: lastAnswerLanguage }),
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Could not generate audio." }));
        throw new Error(err.detail || "Could not generate audio.");
      }
      const blob = await response.blob();
      els.answerAudio.src = URL.createObjectURL(blob);
      els.answerAudio.classList.remove("hidden");
      els.answerAudio.play();
    } catch (err) {
      els.listenBtn.textContent = err.message || "Could not generate audio";
      setTimeout(() => { els.listenBtn.textContent = "🔊 Listen to the answer"; }, 3000);
      els.listenBtn.disabled = false;
      return;
    }
    els.listenBtn.disabled = false;
    els.listenBtn.textContent = "🔊 Listen to the answer";
  }

  els.listenBtn.addEventListener("click", listenToAnswer);

  // -- ask again (same session) / new conversation (reset session) ---------

  function clearInputs() {
    recordedBlob = null;
    recordedChunks = [];
    els.audioPreview.classList.add("hidden");
    els.textInput.value = "";
    els.resultsSection.classList.add("hidden");
    updateSubmitState();
    document.getElementById("intake-slip").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  els.askAgainBtn.addEventListener("click", () => {
    // Keep sessionId — Gemma 4 will remember this conversation's earlier turns.
    clearInputs();
  });

  els.newConvoBtn.addEventListener("click", async () => {
    if (sessionId) {
      fetch(`${API_BASE}/api/conversation/${sessionId}/reset`, { method: "POST" }).catch(() => {});
    }
    sessionId = null;
    els.memoryNote.classList.add("hidden");
    clearInputs();
  });

  // -- init ------------------------------------------------------------------

  function initLocation() {
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          userLat = pos.coords.latitude;
          userLng = pos.coords.longitude;
        },
        (err) => console.warn("Location permission denied or failed.", err)
      );
    }
  }

  function initTimestamp() {
    const now = new Date();
    els.slipTimestamp.textContent = now.toLocaleString(undefined, {
      day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
    });
  }

  loadLanguages();
  initTimestamp();
  initLocation();
  updateSubmitState();
})();
