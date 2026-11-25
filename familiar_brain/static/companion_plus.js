(() => {
  "use strict";

  const el = {
    cameraValue: document.getElementById("camera-value"),
    ambientValue: document.getElementById("ambient-value"),
    btnCamera: document.getElementById("btn-camera"),
    btnCameraSub: document.getElementById("btn-camera-sub"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    chatReply: document.getElementById("chat-reply"),
    btnChatSend: document.getElementById("btn-chat-send"),
    voiceSelect: document.getElementById("voice-select"),
    rollOutput: document.getElementById("roll-output"),
    initForm: document.getElementById("initiative-form"),
    initName: document.getElementById("init-name"),
    initVal: document.getElementById("init-val"),
    initList: document.getElementById("initiative-list"),
    initCurrentTag: document.getElementById("initiative-current-tag"),
    memoryEditor: document.getElementById("memory-editor"),
    feedbackBanner: document.getElementById("feedback-banner"),
    feedbackText: document.getElementById("feedback-text"),
    voiceBootBanner: document.getElementById("voice-boot-banner"),
    voiceBootText: document.getElementById("voice-boot-text"),
    voiceBootStage: document.getElementById("voice-boot-stage"),
    audioLinkTag: document.getElementById("audio-link-tag"),
    micMeterFill: document.getElementById("mic-meter-fill"),
    micMeterPct: document.getElementById("mic-meter-pct"),
    boostInput: document.getElementById("dial-speaker-boost"),
    boostVal: document.getElementById("val-speaker-boost"),
    brandKicker: document.getElementById("brand-kicker"),
    campaignSelect: document.getElementById("campaign-select"),
    campaignRosterCount: document.getElementById("campaign-roster-count"),
    rosterList: document.getElementById("roster-list"),
    rosterForm: document.getElementById("roster-form"),
    rosterInput: document.getElementById("roster-role-input"),
    speakerRoleChips: document.getElementById("speaker-role-chips"),
    campaignCreateForm: document.getElementById("campaign-create-form"),
    newCampaignName: document.getElementById("new-campaign-name"),
    newCampaignSetting: document.getElementById("new-campaign-setting"),
    newCampaignBound: document.getElementById("new-campaign-bound"),
    newCampaignRoster: document.getElementById("new-campaign-roster"),
    newCampaignSecrets: document.getElementById("new-campaign-secrets"),
    privacyList: document.getElementById("privacy-names-list"),
    privacyForm: document.getElementById("privacy-form"),
    privacyInput: document.getElementById("privacy-name-input"),
  };

  const dialNames = ["humor", "honesty", "expressiveness", "curiosity", "bravery"];
  let dialDebounce = null;
  let memoryLoaded = false;

  function notify(msg, tone = "info") {
    if (!el.feedbackBanner || !el.feedbackText) return;
    el.feedbackText.textContent = String(msg || "");
    el.feedbackBanner.classList.remove("hidden", "is-ok", "is-error");
    if (tone === "ok") el.feedbackBanner.classList.add("is-ok");
    if (tone === "error") el.feedbackBanner.classList.add("is-error");
  }

  async function api(url, options = {}) {
    const init = {
      method: options.method || "GET",
      headers: { Accept: "application/json" },
    };
    if (options.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    const resp = await fetch(url, init);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.error || data.detail || `Request failed (${resp.status})`);
    }
    return data;
  }

  const AMBIENT_LABELS = {
    quiet: "Quiet",
    calm_music: "Calm Music",
    tense_music: "Tense Music",
    combat_music: "Combat Music",
    lively_table: "Lively Table",
    sting: "Dramatic Burst",
  };

  function renderPlusState(state) {
    if (!state || typeof state !== "object") return;

    const beh = state.behavior;
    if (beh && typeof beh === "object") {
      for (const name of dialNames) {
        if (typeof beh[name] === "number") {
          const input = document.getElementById(`dial-${name}`);
          const label = document.getElementById(`val-${name}`);
          if (input && document.activeElement !== input) {
            input.value = String(beh[name]);
          }
          if (label) {
            label.textContent = `${beh[name]}%`;
          }
        }
      }
      const open = Boolean(beh.camera_enabled);
      if (el.cameraValue) el.cameraValue.textContent = open ? "Open" : "Closed";
      if (el.btnCamera) {
        el.btnCamera.classList.toggle("is-active", open);
        el.btnCamera.setAttribute("aria-pressed", String(open));
      }
      if (el.btnCameraSub) {
        el.btnCameraSub.textContent = open
          ? "Eyes Open (camera allowed)"
          : "Eyes Closed (table privacy)";
      }
      if (typeof beh.speaker_boost_db === "number") {
        if (el.boostInput && document.activeElement !== el.boostInput) {
          el.boostInput.value = String(beh.speaker_boost_db);
        }
        if (el.boostVal) {
          el.boostVal.textContent = `+${Number(beh.speaker_boost_db).toFixed(1)} dB`;
        }
      }
    }

    if (state.robot && typeof state.robot.mic_level === "number") {
      updateMicMeter(state.robot.mic_level, Boolean(state.robot.backend_connected));
    }

    if (state.ambient && el.ambientValue) {
      const mood = state.ambient.mood || "quiet";
      el.ambientValue.textContent = AMBIENT_LABELS[mood] || mood;
    }

    if (state.voice && el.voiceSelect && document.activeElement !== el.voiceSelect) {
      el.voiceSelect.value = state.voice;
    }

    if (state.initiative) {
      renderInitiative(state.initiative);
    }

    if (state.campaign) {
      renderCampaign(state.campaign, state.campaigns);
    }
  }

  function renderCampaign(active, profiles) {
    if (el.brandKicker && active.name) {
      el.brandKicker.textContent = String(active.name).toUpperCase();
    }
    if (el.campaignSelect && Array.isArray(profiles) && document.activeElement !== el.campaignSelect) {
      const currentKey = profiles.map((p) => `${p.id}:${p.name}`).join("|");
      if (el.campaignSelect.dataset.key !== currentKey) {
        el.campaignSelect.textContent = "";
        for (const p of profiles) {
          const opt = document.createElement("option");
          opt.value = p.id;
          opt.textContent = p.name;
          el.campaignSelect.appendChild(opt);
        }
        el.campaignSelect.dataset.key = currentKey;
      }
      el.campaignSelect.value = active.id || "nephria";
    }

    const roster = Array.isArray(active.roster) ? active.roster : [];
    if (el.campaignRosterCount) {
      el.campaignRosterCount.textContent = `${roster.length} ${roster.length === 1 ? "role" : "roles"}`;
    }
    if (el.rosterList) {
      const key = roster.join("|");
      if (el.rosterList.dataset.key !== key) {
        el.rosterList.textContent = "";
        for (const role of roster) {
          const pill = document.createElement("span");
          pill.className = "role-pill";
          const label = document.createElement("span");
          label.textContent = role;
          const del = document.createElement("button");
          del.type = "button";
          del.className = "role-remove";
          del.dataset.plusAction = "roster_remove";
          del.dataset.role = role;
          del.setAttribute("aria-label", `Remove ${role} from party roster`);
          del.textContent = "\u00d7";
          pill.append(label, del);
          el.rosterList.appendChild(pill);
        }
        el.rosterList.dataset.key = key;
      }
    }

    // Keep Speaker Roles quick-pick chips synced to the active team's live roster
    if (el.speakerRoleChips && roster.length) {
      const chipKey = roster.join("|");
      if (el.speakerRoleChips.dataset.key !== chipKey) {
        el.speakerRoleChips.textContent = "";
        for (const role of roster) {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "chip";
          btn.dataset.action = "pick_role";
          btn.dataset.role = role;
          btn.textContent = role;
          el.speakerRoleChips.appendChild(btn);
        }
        el.speakerRoleChips.dataset.key = chipKey;
      }
    }

    const priv = Array.isArray(active.privacy_names) ? active.privacy_names : [];
    if (el.privacyList) {
      const pKey = priv.join("|");
      if (el.privacyList.dataset.key !== pKey) {
        el.privacyList.textContent = "";
        if (!priv.length) {
          const p = document.createElement("p");
          p.className = "empty-state";
          p.textContent = "No real player names added for this team.";
          el.privacyList.appendChild(p);
        } else {
          for (const name of priv) {
            const pill = document.createElement("span");
            pill.className = "role-pill";
            const label = document.createElement("span");
            label.textContent = name;
            const del = document.createElement("button");
            del.type = "button";
            del.className = "role-remove";
            del.dataset.plusAction = "privacy_remove";
            del.dataset.name = name;
            del.setAttribute("aria-label", `Remove ${name} from privacy filter`);
            del.textContent = "\u00d7";
            pill.append(label, del);
            el.privacyList.appendChild(pill);
          }
        }
        el.privacyList.dataset.key = pKey;
      }
    }
  }

  function renderInitiative(init) {
    if (!el.initList) return;
    const order = Array.isArray(init.order) ? init.order : [];
    const current = init.current || null;
    if (el.initCurrentTag) {
      el.initCurrentTag.textContent = current ? `Up: ${current}` : "No turn active";
    }
    el.initList.textContent = "";
    if (!order.length) {
      const p = document.createElement("p");
      p.className = "empty-state";
      p.textContent = "Combat order is empty.";
      el.initList.appendChild(p);
      return;
    }
    for (const row of order) {
      const div = document.createElement("div");
      div.className = "enrolled-row" + (row.name === current ? " init-row-active" : "");
      const span = document.createElement("span");
      span.className = "mono";
      span.textContent = `${row.name === current ? "▶ " : ""}${row.name} (${row.value})`;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-link";
      btn.textContent = "Remove";
      btn.dataset.plusAction = "init_remove";
      btn.dataset.name = row.name;
      div.append(span, btn);
      el.initList.appendChild(div);
    }
  }

  async function syncState() {
    try {
      const state = await api("/api/state");
      renderPlusState(state);
    } catch (_) {
      // app.js handles offline status banner
    }
  }

  async function loadMemory(quiet = false) {
    if (!el.memoryEditor) return;
    try {
      const data = await api("/api/memory");
      el.memoryEditor.value = data.campaign_memory || "";
      memoryLoaded = true;
      if (!quiet) notify("Loaded campaign memory.", "ok");
    } catch (err) {
      if (!quiet) notify(err.message, "error");
    }
  }

  async function saveMemory() {
    if (!el.memoryEditor) return;
    try {
      const res = await api("/api/memory", {
        method: "PUT",
        body: { campaign_memory: el.memoryEditor.value },
      });
      notify(`Saved and reindexed ${res.indexed_chunks} memory chunk(s).`, "ok");
    } catch (err) {
      notify(err.message, "error");
    }
  }

  // Dial slider changes (debounced POST /api/behavior)
  for (const name of dialNames) {
    const input = document.getElementById(`dial-${name}`);
    const label = document.getElementById(`val-${name}`);
    if (!input) continue;
    input.addEventListener("input", () => {
      if (label) label.textContent = `${input.value}%`;
      clearTimeout(dialDebounce);
      dialDebounce = setTimeout(async () => {
        try {
          const res = await api("/api/behavior", {
            method: "POST",
            body: { [name]: Number(input.value) },
          });
          renderPlusState({ behavior: res });
          notify(`${name.charAt(0).toUpperCase() + name.slice(1)} set to ${res[name]}%.`, "ok");
        } catch (err) {
          notify(err.message, "error");
        }
      }, 180);
    });
  }

  function updateMicMeter(level, connected) {
    const pct = Math.max(0, Math.min(100, Math.round((Number(level) || 0) * 100)));
    if (el.micMeterFill) el.micMeterFill.style.width = `${pct}%`;
    if (el.micMeterPct) el.micMeterPct.textContent = `${pct}%`;
    if (el.audioLinkTag && connected !== undefined) {
      el.audioLinkTag.textContent = connected ? "audio live" : "audio idle";
    }
  }

  async function pollRobotBootStatus() {
    try {
      const resp = await fetch("/_familiar/boot-status", { cache: "no-store" });
      if (!resp.ok) return;
      const d = await resp.json();
      if (el.voiceBootBanner) {
        const warming = d.voice_ready === false;
        el.voiceBootBanner.classList.toggle("hidden", !warming);
        if (warming) {
          if (el.voiceBootText) el.voiceBootText.textContent = d.detail || "Voice engine warming up...";
          if (el.voiceBootStage) el.voiceBootStage.textContent = String(d.stage || "warming").toUpperCase();
        }
      }
      if (typeof d.mic_level === "number") {
        updateMicMeter(d.mic_level, Boolean(d.audio_connected));
      } else if (el.audioLinkTag && d.voice_ready === false) {
        el.audioLinkTag.textContent = "warming up...";
      }
    } catch (_) {
      // Running directly against laptop :8090 without Reachy :8043 proxy
    }
  }

  if (el.boostInput) {
    let boostTimer = null;
    el.boostInput.addEventListener("input", () => {
      const val = Number(el.boostInput.value);
      if (el.boostVal) el.boostVal.textContent = `+${val.toFixed(1)} dB`;
      clearTimeout(boostTimer);
      boostTimer = setTimeout(async () => {
        try {
          const res = await api("/api/behavior", {
            method: "POST",
            body: { speaker_boost_db: val },
          });
          renderPlusState({ behavior: res });
          notify(`Speaker boost set to +${Number(res.speaker_boost_db).toFixed(1)} dB.`, "ok");
        } catch (err) {
          notify(err.message, "error");
        }
      }, 180);
    });
  }

  // Click actions for Phase 2 controls
  document.addEventListener("click", async (ev) => {
    const btn = ev.target.closest("[data-plus-action]");
    if (!btn) return;
    const action = btn.dataset.plusAction;
    try {
      if (action === "toggle_camera") {
        const res = await api("/api/camera", { method: "POST", body: {} });
        await syncState();
        notify(res.camera_enabled ? "Eyes opened (camera enabled)." : "Eyes closed (camera disabled).", "ok");
      } else if (action === "apply_preset") {
        const preset = btn.dataset.preset;
        const res = await api("/api/behavior", { method: "POST", body: { preset } });
        renderPlusState({ behavior: res });
        notify(`Applied ${preset.replace("_", " ")} preset.`, "ok");
      } else if (action === "quick_roll") {
        const res = await api("/api/roll", {
          method: "POST",
          body: { expression: btn.dataset.expr, advantage: btn.dataset.adv || "none" },
        });
        if (el.rollOutput) {
          el.rollOutput.textContent = res.result;
          el.rollOutput.classList.remove("hidden");
        }
      } else if (action === "init_next") {
        const res = await api("/api/initiative", { method: "POST", body: { action: "next" } });
        renderInitiative(res);
        notify(res.result, "ok");
      } else if (action === "init_clear") {
        const res = await api("/api/initiative", { method: "POST", body: { action: "clear" } });
        renderInitiative(res);
        notify(res.result, "ok");
      } else if (action === "init_remove") {
        const res = await api("/api/initiative", {
          method: "POST",
          body: { action: "remove", name: btn.dataset.name },
        });
        renderInitiative(res);
      } else if (action === "load_memory") {
        await loadMemory(false);
      } else if (action === "save_memory") {
        await saveMemory();
      } else if (action === "speak_test") {
        const res = await api("/_familiar/speak-test", { method: "POST", body: {} });
        notify(res.detail || "Played speaker test on Reachy Mini.", "ok");
      } else if (action === "reset_audio") {
        const res = await api("/_familiar/reconnect-audio", { method: "POST", body: {} });
        notify(res.detail || "Restarting Reachy audio link...", "ok");
      } else if (action === "roster_remove") {
        const res = await api("/api/roster", {
          method: "POST",
          body: { action: "remove", role: btn.dataset.role },
        });
        await syncState();
        notify(`Removed ${btn.dataset.role} from party roster (${res.roster.length} remaining).`, "ok");
      } else if (action === "privacy_remove") {
        await api("/api/privacy-names", {
          method: "POST",
          body: { action: "remove", name: btn.dataset.name },
        });
        await syncState();
        notify(`Removed ${btn.dataset.name} from privacy filter.`, "ok");
      }
    } catch (err) {
      notify(err.message, "error");
    }
  });

  if (el.initForm) {
    el.initForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const name = (el.initName.value || "").trim();
      const val = Number(el.initVal.value);
      if (!name || !Number.isFinite(val)) {
        notify("Enter a combatant name and initiative number.", "error");
        return;
      }
      try {
        const res = await api("/api/initiative", {
          method: "POST",
          body: { action: "add", name, value: Math.round(val) },
        });
        el.initName.value = "";
        el.initVal.value = "";
        renderInitiative(res);
      } catch (err) {
        notify(err.message, "error");
      }
    });
  }

  if (el.chatForm) {
    el.chatForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const text = (el.chatInput.value || "").trim();
      if (!text) return;
      el.btnChatSend.disabled = true;
      try {
        const res = await api("/api/chat", { method: "POST", body: { text } });
        el.chatInput.value = "";
        if (el.chatReply) {
          el.chatReply.textContent = res.reply;
          el.chatReply.classList.remove("hidden");
        }
      } catch (err) {
        notify(err.message, "error");
      } finally {
        el.btnChatSend.disabled = false;
      }
    });
  }

  if (el.voiceSelect) {
    el.voiceSelect.addEventListener("change", async () => {
      try {
        const res = await api("/api/voice", {
          method: "POST",
          body: { voice: el.voiceSelect.value },
        });
        notify(`Kokoro voice switched to ${res.voice} (takes effect on next reply).`, "ok");
      } catch (err) {
        notify(err.message, "error");
      }
    });
  }

  if (el.campaignSelect) {
    el.campaignSelect.addEventListener("change", async () => {
      try {
        const res = await api("/api/campaigns/switch", {
          method: "POST",
          body: { id: el.campaignSelect.value },
        });
        await syncState();
        await loadMemory(true);
        window.dispatchEvent(new CustomEvent("familiar:campaign-switched", { detail: res.active }));
        notify(`Switched team profile to ${res.active.name}.`, "ok");
      } catch (err) {
        notify(err.message, "error");
      }
    });
  }

  if (el.rosterForm) {
    el.rosterForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const role = (el.rosterInput.value || "").trim();
      if (!role) return;
      try {
        await api("/api/roster", { method: "POST", body: { action: "add", role } });
        el.rosterInput.value = "";
        await syncState();
        notify(`Added ${role} to party roster.`, "ok");
      } catch (err) {
        notify(err.message, "error");
      }
    });
  }

  if (el.privacyForm) {
    el.privacyForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const name = (el.privacyInput.value || "").trim();
      if (!name) return;
      try {
        await api("/api/privacy-names", { method: "POST", body: { action: "add", name } });
        el.privacyInput.value = "";
        await syncState();
        notify(`Added ${name} to real-name privacy filter.`, "ok");
      } catch (err) {
        notify(err.message, "error");
      }
    });
  }

  if (el.campaignCreateForm) {
    el.campaignCreateForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const name = (el.newCampaignName.value || "").trim();
      if (!name) return;
      try {
        const created = await api("/api/campaigns", {
          method: "POST",
          body: {
            name,
            setting: (el.newCampaignSetting.value || "").trim(),
            bound_to: (el.newCampaignBound.value || "").trim(),
            roster: (el.newCampaignRoster.value || "").trim(),
            secrets: (el.newCampaignSecrets.value || "").trim(),
          },
        });
        const sw = await api("/api/campaigns/switch", {
          method: "POST",
          body: { id: created.profile.id },
        });
        el.campaignCreateForm.reset();
        await syncState();
        await loadMemory(true);
        window.dispatchEvent(new CustomEvent("familiar:campaign-switched", { detail: sw.active }));
        notify(`Created and switched to team profile "${sw.active.name}".`, "ok");
      } catch (err) {
        notify(err.message, "error");
      }
    });
  }

  // Instant SSE live updates with automatic fallback to periodic sync
  if (typeof window.EventSource === "function") {
    try {
      const es = new EventSource("/api/events");
      es.addEventListener("state", (ev) => {
        try {
          const parsed = JSON.parse(ev.data);
          renderPlusState(parsed);
          window.dispatchEvent(new CustomEvent("familiar:state", { detail: parsed }));
        } catch (_) {
          // ignore malformed event
        }
      });
    } catch (_) {
      // fallback polling below
    }
  }

  syncState().then(() => {
    if (!memoryLoaded) loadMemory(true);
  });
  pollRobotBootStatus();
  setInterval(syncState, 4000);
  setInterval(pollRobotBootStatus, 750);
})();
