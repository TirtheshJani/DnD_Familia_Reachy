(() => {
  "use strict";

  const POLL_MS = 2000;
  const CONFIRM_RESET_MS = 4000;

  const ui = {
    statusPill: document.getElementById("primary-status-pill"),
    statusIcon: document.getElementById("primary-status-icon"),
    statusText: document.getElementById("primary-status-text"),
    sessionTimer: document.getElementById("session-timer"),
    latencyValue: document.getElementById("latency-value"),
    robotBadge: document.getElementById("robot-badge"),
    speakerBadge: document.getElementById("speaker-badge"),
    modeValue: document.getElementById("mode-value"),
    trackingValue: document.getElementById("tracking-value"),
    loggingValue: document.getElementById("logging-value"),
    sessionIdTag: document.getElementById("session-id-tag"),
    feedbackBanner: document.getElementById("feedback-banner"),
    feedbackText: document.getElementById("feedback-text"),
    btnSpeakNow: document.getElementById("btn-speak-now"),
    btnMute: document.getElementById("btn-mute"),
    btnUnmute: document.getElementById("btn-unmute"),
    btnMuteStop: document.getElementById("btn-mute-stop"),
    btnMode: document.getElementById("btn-mode"),
    btnModeSub: document.getElementById("btn-mode-sub"),
    btnDeepNight: document.getElementById("btn-deep-night"),
    btnDeepNightSub: document.getElementById("btn-deep-night-sub"),
    noteForm: document.getElementById("note-form"),
    noteInput: document.getElementById("note-input"),
    btnCheckpoint: document.getElementById("btn-checkpoint"),
    btnEndSession: document.getElementById("btn-end-session"),
    sessionList: document.getElementById("session-list"),
    robotHostTag: document.getElementById("robot-host-tag"),
    robotDetail: document.getElementById("robot-detail"),
    speakerForm: document.getElementById("speaker-form"),
    speakerRoleInput: document.getElementById("speaker-role-input"),
    btnEnrollSpeaker: document.getElementById("btn-enroll-speaker"),
    btnClearSpeakers: document.getElementById("btn-clear-speakers"),
    speakersEnabledTag: document.getElementById("speakers-enabled-tag"),
    enrolledSpeakers: document.getElementById("enrolled-speakers"),
    transcriptCount: document.getElementById("transcript-count"),
    transcriptList: document.getElementById("transcript-list"),
  };

  let latestState = null;
  let pollInFlight = false;
  let brainErrorShown = false;
  let pendingPurgeSid = null;
  let purgeTimer = null;
  let clearArmed = false;
  let clearTimer = null;
  let enrollCountdownTimer = null;

  function showBanner(message, tone = "info") {
    if (!ui.feedbackBanner || !ui.feedbackText) return;
    ui.feedbackText.textContent = String(message || "");
    ui.feedbackBanner.classList.remove("hidden", "is-ok", "is-error");
    if (tone === "ok") ui.feedbackBanner.classList.add("is-ok");
    if (tone === "error") ui.feedbackBanner.classList.add("is-error");
  }

  function hideBanner() {
    if (!ui.feedbackBanner) return;
    ui.feedbackBanner.classList.add("hidden");
  }

  function formatDuration(seconds) {
    const total = Math.max(0, Math.floor(seconds));
    const hrs = String(Math.floor(total / 3600)).padStart(2, "0");
    const mins = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
    const secs = String(total % 60).padStart(2, "0");
    return `${hrs}:${mins}:${secs}`;
  }

  function formatClock(epochSeconds) {
    if (typeof epochSeconds !== "number" || !Number.isFinite(epochSeconds)) {
      return "--:--:--";
    }
    const d = new Date(epochSeconds * 1000);
    return d.toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  async function apiJson(url, options = {}) {
    const init = {
      method: options.method || "GET",
      headers: { Accept: "application/json" },
    };
    if (options.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    // A hung request would leave pollInFlight set forever and freeze every later poll.
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    init.signal = controller.signal;
    let resp;
    try {
      resp = await fetch(url, init);
    } catch (err) {
      if (err && err.name === "AbortError") throw new Error("Request timed out after 5 s");
      throw err;
    } finally {
      clearTimeout(timer);
    }
    let data = {};
    try {
      data = await resp.json();
    } catch (_) {
      data = {};
    }
    if (!resp.ok) {
      const errMsg = data.error || data.detail || `Request failed (${resp.status})`;
      const err = new Error(errMsg);
      err.status = resp.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  function updateSessionTimer() {
    if (!ui.sessionTimer) return;
    const started = latestState && latestState.session_started_at;
    if (latestState && latestState.session && typeof started === "number") {
      ui.sessionTimer.textContent = formatDuration((Date.now() / 1000) - started);
    } else if (latestState && latestState.session) {
      ui.sessionTimer.textContent = "Active";
    } else {
      ui.sessionTimer.textContent = "Idle";
    }
  }

  function renderStatus(state, offline = false) {
    if (offline) {
      ui.statusPill.className = "status-pill state-offline";
      ui.statusIcon.textContent = "!";
      ui.statusText.textContent = "OFFLINE";
      return;
    }

    let pillClass = "state-listening";
    let icon = "\u25c6";
    let label = "LISTENING";

    if (state.muted && !state.logging) {
      pillClass = "state-muted";
      icon = "\u25a0";
      label = "MUTED \u00b7 NO LOG";
    } else if (state.muted) {
      pillClass = "state-muted";
      icon = "\u25a0";
      label = "MUTED";
    } else if (state.asleep) {
      pillClass = "state-night";
      icon = "\u263d";
      label = "DEEP NIGHT";
    } else if (state.mode === "thinking") {
      pillClass = "state-thinking";
      icon = "\u2726";
      label = "THINKING";
    }

    ui.statusPill.className = `status-pill ${pillClass}`;
    ui.statusIcon.textContent = icon;
    ui.statusText.textContent = label;

    updateSessionTimer();

    if (typeof state.last_turn_ms === "number" && Number.isFinite(state.last_turn_ms)) {
      ui.latencyValue.textContent = `${Math.round(state.last_turn_ms)} ms`;
    } else {
      ui.latencyValue.textContent = "\u2014";
    }

    const robot = state.robot || {};
    if (robot.backend_connected) {
      ui.robotBadge.textContent = "Connected";
      ui.robotHostTag.textContent = robot.host || "linked";
    } else if (robot.disabled) {
      ui.robotBadge.textContent = "Disabled";
      ui.robotHostTag.textContent = "local-only";
    } else {
      ui.robotBadge.textContent = "Offline";
      ui.robotHostTag.textContent = robot.host || "unlinked";
    }
    ui.robotDetail.textContent = robot.detail || "Robot status unavailable.";

    const speakers = state.speakers || {};
    if (speakers.last_speaker) {
      const pct = typeof speakers.last_confidence === "number"
        ? ` (${Math.round(speakers.last_confidence * 100)}%)`
        : "";
      ui.speakerBadge.textContent = `${speakers.last_speaker}${pct}`;
    } else {
      ui.speakerBadge.textContent = "\u2014";
    }

    const isThinking = state.mode === "thinking";
    ui.modeValue.textContent = isThinking ? "Thinking" : "Familiar";
    ui.trackingValue.textContent = state.tracking ? "On" : "Off";
    ui.loggingValue.textContent = state.logging ? "On" : "Stopped";
    ui.sessionIdTag.textContent = state.session ? `Session ${state.session}` : "No active session";

    ui.btnSpeakNow.disabled = Boolean(state.muted || state.asleep);
    ui.btnEndSession.disabled = !state.session;
    ui.btnCheckpoint.disabled = !state.session;

    ui.btnMode.classList.toggle("is-active", isThinking);
    ui.btnMode.setAttribute("aria-pressed", String(isThinking));
    ui.btnModeSub.textContent = isThinking ? "Thinking (6 sentences)" : "Familiar (2 sentences)";

    const isAsleep = Boolean(state.asleep);
    ui.btnDeepNight.classList.toggle("is-active", isAsleep);
    ui.btnDeepNight.setAttribute("aria-pressed", String(isAsleep));
    ui.btnDeepNightSub.textContent = isAsleep ? "Asleep (Quiet)" : "Awake";

    renderSpeakers(speakers);
    renderTranscript(state.transcript || []);
  }

  function renderSpeakers(speakers) {
    if (!ui.enrolledSpeakers) return;
    const enabled = Boolean(speakers.enabled);
    ui.speakersEnabledTag.textContent = enabled ? "Opt-in CPU active" : "Disabled in config";

    const roles = Array.isArray(speakers.enrolled) ? speakers.enrolled : [];
    if (roles.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "No voice profiles enrolled. Rocky treats all speakers equally.";
      ui.enrolledSpeakers.replaceChildren(empty);
      return;
    }

    const nodes = roles.map((role) => {
      const pill = document.createElement("span");
      pill.className = "role-pill";

      const label = document.createElement("span");
      label.textContent = String(role);

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "role-remove";
      delBtn.dataset.action = "delete_speaker";
      delBtn.dataset.role = String(role);
      delBtn.setAttribute("aria-label", `Remove speaker role ${role}`);
      delBtn.textContent = "\u00d7";

      pill.append(label, delBtn);
      return pill;
    });
    ui.enrolledSpeakers.replaceChildren(...nodes);
  }

  function renderTranscript(entries) {
    if (!ui.transcriptList) return;
    const rows = Array.isArray(entries) ? entries : [];
    ui.transcriptCount.textContent = `${rows.length} ${rows.length === 1 ? "turn" : "turns"}`;

    if (rows.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "No utterances recorded yet. Speak at the table or start a session.";
      ui.transcriptList.replaceChildren(empty);
      return;
    }

    const items = rows.slice().reverse().map((entry) => {
      const card = document.createElement("article");
      card.className = `transcript-item${entry.spoke ? " did-speak" : ""}`;

      const meta = document.createElement("div");
      meta.className = "transcript-meta";

      const speakerTag = document.createElement("span");
      speakerTag.className = "speaker-tag";
      speakerTag.textContent = entry.speaker ? String(entry.speaker) : "Table";

      const decisionTag = document.createElement("span");
      decisionTag.className = "decision-tag";
      decisionTag.textContent = String(entry.decision || "IGNORE");

      meta.append(speakerTag, decisionTag);

      if (entry.spoke) {
        const spokeTag = document.createElement("span");
        spokeTag.className = "spoke-tag";
        spokeTag.textContent = "Spoke";
        meta.append(spokeTag);
      }

      const timeTag = document.createElement("span");
      timeTag.className = "time-tag";
      timeTag.textContent = formatClock(entry.ts);
      meta.append(timeTag);

      const body = document.createElement("p");
      body.className = "transcript-text";
      body.textContent = String(entry.text || "");

      card.append(meta, body);
      return card;
    });

    ui.transcriptList.replaceChildren(...items);
  }

  function renderSessions(sessions) {
    if (!ui.sessionList) return;
    const list = Array.isArray(sessions) ? sessions : [];
    if (list.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "No saved sessions found.";
      ui.sessionList.replaceChildren(empty);
      return;
    }

    const activeId = latestState ? latestState.session : null;
    const rows = list.slice().reverse().map((sess) => {
      const sid = String(sess.id || "");
      const row = document.createElement("div");
      row.className = "session-row";

      const info = document.createElement("div");
      info.className = "session-info";

      const idEl = document.createElement("span");
      idEl.className = "session-id";
      idEl.textContent = sid;

      const flagsEl = document.createElement("span");
      flagsEl.className = "session-flags";
      const flags = [];
      if (sid === activeId) flags.push("ACTIVE");
      if (sess.raw) flags.push("raw");
      if (sess.summary) flags.push("summary");
      flagsEl.textContent = flags.join(" \u00b7 ") || "log";

      info.append(idEl, flagsEl);

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-danger-outline btn-compact";
      btn.dataset.action = "purge_session";
      btn.dataset.sid = sid;
      if (sid === activeId) {
        btn.disabled = true;
        btn.textContent = "Active";
      } else if (pendingPurgeSid === sid) {
        btn.textContent = "Confirm Purge";
      } else {
        btn.textContent = "Purge";
      }

      row.append(info, btn);
      return row;
    });

    ui.sessionList.replaceChildren(...rows);
  }

  function applyState(s) {
    latestState = s;
    renderStatus(s, false);
  }

  async function refreshState() {
    if (pollInFlight) return;
    pollInFlight = true;
    try {
      const state = await apiJson("/api/state");
      applyState(state);
      if (brainErrorShown) {
        brainErrorShown = false;
        hideBanner();
      }
    } catch (err) {
      renderStatus(latestState || {}, true);
      showBanner(`Brain unreachable: ${err.message}`, "error");
      brainErrorShown = true;
    } finally {
      pollInFlight = false;
    }
  }

  async function refreshSessions() {
    try {
      const data = await apiJson("/api/sessions");
      renderSessions(data.sessions || []);
    } catch (err) {
      showBanner(`Could not list sessions: ${err.message}`, "error");
    }
  }

  async function runWithBusy(button, fn) {
    if (button) button.disabled = true;
    try {
      await fn();
    } finally {
      if (button) button.disabled = false;
      if (latestState) renderStatus(latestState, false);
    }
  }

  async function handleAction(action, target) {
    switch (action) {
      case "dismiss_banner":
        hideBanner();
        return;

      case "mute":
        await apiJson("/api/mute", { method: "POST", body: {} });
        await refreshState();
        showBanner("Rocky muted (still logging).", "ok");
        return;

      case "unmute":
        await apiJson("/api/unmute", { method: "POST", body: {} });
        await refreshState();
        showBanner("Rocky unmuted and listening.", "ok");
        return;

      case "mute_and_stop_logging":
        await apiJson("/api/mute_and_stop_logging", { method: "POST", body: {} });
        await refreshState();
        showBanner("Rocky muted and logging stopped.", "ok");
        return;

      case "speak_now":
        try {
          await apiJson("/api/speak_now", { method: "POST", body: {} });
          showBanner("Speak Now armed for the next table line.", "ok");
        } catch (err) {
          showBanner(err.message, "error");
        }
        await refreshState();
        return;

      case "toggle_mode": {
        const nextMode = latestState && latestState.mode === "thinking" ? "familiar" : "thinking";
        const res = await apiJson("/api/mode", { method: "POST", body: { mode: nextMode } });
        await refreshState();
        showBanner(`Mode switched to ${res.mode}.`, "ok");
        return;
      }

      case "toggle_deep_night": {
        const nextAsleep = !(latestState && latestState.asleep);
        const res = await apiJson("/api/deep_night", { method: "POST", body: { asleep: nextAsleep } });
        await refreshState();
        showBanner(res.asleep ? "Deep Night enabled." : "Deep Night ended.", "ok");
        return;
      }

      case "send_note": {
        const text = (ui.noteInput.value || "").trim();
        if (!text) {
          showBanner("Enter a note before sending.", "error");
          return;
        }
        const res = await apiJson("/api/note", { method: "POST", body: { text } });
        ui.noteInput.value = "";
        await refreshState();
        showBanner(res.result || "Note recorded.", "ok");
        return;
      }

      case "start_session": {
        const res = await apiJson("/api/session/start", { method: "POST", body: {} });
        await refreshState();
        await refreshSessions();
        showBanner(`Started session ${res.session}.`, "ok");
        return;
      }

      case "checkpoint":
        showBanner("Running break checkpoint summary...", "info");
        try {
          const res = await apiJson("/api/session/checkpoint", { method: "POST", body: { force: true } });
          await refreshState();
          await refreshSessions();
          const msg = res.campaign_updated
            ? "Break checkpoint saved and campaign memory refreshed."
            : "Break checkpoint complete (no new scenes to fold yet).";
          showBanner(msg, "ok");
        } catch (err) {
          showBanner(err.message, "error");
        }
        return;

      case "end_session":
        try {
          await apiJson("/api/session/end", { method: "POST", body: {} });
          await refreshState();
          await refreshSessions();
          showBanner("Session ended and queued for summary.", "ok");
        } catch (err) {
          showBanner(err.message, "error");
        }
        return;

      case "reindex": {
        const res = await apiJson("/api/reindex", { method: "POST", body: {} });
        showBanner(`Notes reindexed: ${res.indexed || 0} updated, ${res.skipped || 0} unchanged.`, "ok");
        return;
      }

      case "refresh_sessions":
        await refreshSessions();
        return;

      case "purge_session": {
        const sid = target && target.dataset ? target.dataset.sid : "";
        if (!sid) return;
        if (pendingPurgeSid !== sid) {
          pendingPurgeSid = sid;
          clearTimeout(purgeTimer);
          purgeTimer = setTimeout(() => {
            pendingPurgeSid = null;
            refreshSessions();
          }, CONFIRM_RESET_MS);
          await refreshSessions();
          showBanner(`Tap "Confirm Purge" to delete session ${sid}.`, "info");
          return;
        }
        clearTimeout(purgeTimer);
        pendingPurgeSid = null;
        try {
          await apiJson(`/api/session/${encodeURIComponent(sid)}`, { method: "DELETE" });
          await refreshSessions();
          showBanner(`Purged session ${sid}.`, "ok");
        } catch (err) {
          showBanner(err.message, "error");
        }
        return;
      }

      case "reconnect_robot":
        showBanner("Reconnecting to Reachy Mini...", "info");
        try {
          let res;
          if (latestState && latestState.robot && latestState.robot.mode === "native") {
            try {
              res = await apiJson("/_familiar/reconnect-audio", { method: "POST", body: {} });
            } catch (_) {
              res = await apiJson("/api/robot/reconnect", { method: "POST", body: {} });
            }
          } else {
            res = await apiJson("/api/robot/reconnect", { method: "POST", body: {} });
          }
          await refreshState();
          showBanner(res.detail || "Robot connected.", "ok");
        } catch (err) {
          await refreshState();
          showBanner(err.message, "error");
        }
        return;

      case "refresh_robot":
        try {
          const res = await apiJson("/api/robot/status");
          await refreshState();
          showBanner(res.detail || "Robot status updated.", res.backend_connected ? "ok" : "info");
        } catch (err) {
          showBanner(err.message, "error");
        }
        return;

      case "pick_role": {
        const role = target && target.dataset ? target.dataset.role : "";
        if (role && ui.speakerRoleInput) {
          ui.speakerRoleInput.value = role;
          ui.speakerRoleInput.focus();
        }
        return;
      }

      case "enroll_speaker": {
        const role = (ui.speakerRoleInput.value || "").trim();
        if (!role) {
          showBanner("Enter a character or role name (e.g. DM, Brokk) first.", "error");
          return;
        }
        let remaining = 5;
        ui.btnEnrollSpeaker.textContent = `Listening (${remaining}s)...`;
        clearInterval(enrollCountdownTimer);
        enrollCountdownTimer = setInterval(() => {
          remaining = Math.max(0, remaining - 1);
          ui.btnEnrollSpeaker.textContent = remaining > 0 ? `Listening (${remaining}s)...` : "Processing...";
        }, 1000);
        try {
          const res = await apiJson("/api/speakers/enroll", {
            method: "POST",
            body: { role, duration_s: 5 },
          });
          ui.speakerRoleInput.value = "";
          await refreshState();
          showBanner(`Enrolled voice profile for ${res.role}.`, "ok");
        } catch (err) {
          showBanner(err.message, "error");
        } finally {
          clearInterval(enrollCountdownTimer);
          ui.btnEnrollSpeaker.textContent = "Enroll (5s)";
        }
        return;
      }

      case "delete_speaker": {
        const role = target && target.dataset ? target.dataset.role : "";
        if (!role) return;
        await apiJson(`/api/speakers/${encodeURIComponent(role)}`, { method: "DELETE" });
        await refreshState();
        showBanner(`Removed speaker profile ${role}.`, "ok");
        return;
      }

      case "clear_speakers":
        if (!clearArmed) {
          clearArmed = true;
          ui.btnClearSpeakers.textContent = "Tap Again to Clear All";
          clearTimeout(clearTimer);
          clearTimer = setTimeout(() => {
            clearArmed = false;
            ui.btnClearSpeakers.textContent = "Clear All Profiles";
          }, CONFIRM_RESET_MS);
          return;
        }
        clearTimeout(clearTimer);
        clearArmed = false;
        ui.btnClearSpeakers.textContent = "Clear All Profiles";
        await apiJson("/api/speakers", { method: "DELETE" });
        await refreshState();
        showBanner("Cleared all speaker profiles.", "ok");
        return;

      default:
        return;
    }
  }

  document.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-action]");
    if (!btn) return;
    if (btn.type === "submit") return;
    const action = btn.dataset.action;
    runWithBusy(btn, () => handleAction(action, btn)).catch((err) => {
      showBanner(err.message || "Action failed.", "error");
    });
  });

  if (ui.noteForm) {
    ui.noteForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const submitBtn = ui.noteForm.querySelector("[data-action='send_note']");
      runWithBusy(submitBtn, () => handleAction("send_note", submitBtn)).catch((err) => {
        showBanner(err.message || "Failed to send note.", "error");
      });
    });
  }

  if (ui.speakerForm) {
    ui.speakerForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      runWithBusy(ui.btnEnrollSpeaker, () => handleAction("enroll_speaker", ui.btnEnrollSpeaker)).catch((err) => {
        showBanner(err.message || "Enrollment failed.", "error");
      });
    });
  }

  window.addEventListener("familiar:state", (ev) => {
    if (ev && ev.detail && typeof ev.detail === "object") {
      applyState(ev.detail);
    }
  });
  window.addEventListener("familiar:campaign-switched", () => {
    refreshState().then(refreshSessions);
  });

  refreshState().then(refreshSessions);
  setInterval(refreshState, POLL_MS);
  setInterval(updateSessionTimer, 1000);
})();
