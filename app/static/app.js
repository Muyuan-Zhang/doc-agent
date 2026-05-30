// Session identity — persisted across page reloads
const SESSION_ID = (() => {
  const stored = localStorage.getItem("session_id");
  if (stored) return stored;
  const id = crypto.randomUUID();
  localStorage.setItem("session_id", id);
  return id;
})();

// DOM references
const dropZone    = document.getElementById("drop-zone");
const fileInput   = document.getElementById("file-input");
const fileList    = document.getElementById("file-list");
const messages    = document.getElementById("messages");
const queryInput  = document.getElementById("query-input");
const sendBtn     = document.getElementById("send-btn");

// ── Toast ────────────────────────────────────────────────────────────────────

let _toastEl = null;
function showToast(text, durationMs = 3500) {
  if (!_toastEl) {
    _toastEl = document.createElement("div");
    _toastEl.className = "toast";
    document.body.appendChild(_toastEl);
  }
  _toastEl.textContent = text;
  _toastEl.classList.add("show");
  setTimeout(() => _toastEl.classList.remove("show"), durationMs);
}

// ── File upload ──────────────────────────────────────────────────────────────

// Map of doc_id → { name, status }
const uploadedFiles = new Map();

function renderFileItem(name) {
  const li = document.createElement("li");
  li.className = "file-item";

  const nameSpan = document.createElement("span");
  nameSpan.className = "file-name";
  nameSpan.title = name;
  nameSpan.textContent = name;

  const badge = document.createElement("span");
  badge.className = "badge badge-ingesting";
  badge.textContent = "ingesting";

  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.className = "delete-btn";
  deleteBtn.textContent = "×";
  deleteBtn.title = "Remove document";

  li.appendChild(nameSpan);
  li.appendChild(badge);
  li.appendChild(deleteBtn);
  fileList.appendChild(li);
  return { badge, li, deleteBtn };
}

function updateBadge(badge, status) {
  badge.className = `badge badge-${status}`;
  badge.textContent = status;
}

async function deleteDocument(docId, li, deleteBtn) {
  deleteBtn.disabled = true;
  try {
    const res = await fetch(`/knowledge-base/documents/${docId}`, { method: "DELETE" });
    if (res.status === 204 || res.ok) {
      uploadedFiles.delete(docId);
      li.remove();
    } else {
      showToast(`Delete failed (${res.status})`);
      deleteBtn.disabled = false;
    }
  } catch (err) {
    showToast("Delete error: " + err.message);
    deleteBtn.disabled = false;
  }
}

// Backend writes "indexed" on success, "failed" on error (DocumentStatus literal).
// Cap at 150 attempts (~5 min at 2 s each) to avoid an infinite loop on stuck jobs.
async function pollDocStatus(docId, badge, maxAttempts = 150) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    await new Promise(r => setTimeout(r, 2000));
    if (!uploadedFiles.has(docId)) return;  // document was deleted during polling
    try {
      const res = await fetch(`/knowledge-base/documents/${docId}/status`);
      if (!res.ok) {
        updateBadge(badge, "error");
        uploadedFiles.get(docId).status = "error";
        return;
      }
      const { status } = await res.json();
      if (status === "indexed" || status === "ready" || status === "done") {
        updateBadge(badge, "ready");
        uploadedFiles.get(docId).status = "ready";
        return;
      }
      if (status === "failed" || status === "error") {
        updateBadge(badge, "error");
        uploadedFiles.get(docId).status = "error";
        return;
      }
      updateBadge(badge, "ingesting");
    } catch {
      updateBadge(badge, "error");
      return;
    }
  }
  // Timed out waiting for ingestion
  updateBadge(badge, "error");
  if (uploadedFiles.has(docId)) uploadedFiles.get(docId).status = "error";
}

async function uploadFile(file) {
  const { badge, li, deleteBtn } = renderFileItem(file.name);
  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/knowledge-base/documents", { method: "POST", body: form });
    if (!res.ok) {
      updateBadge(badge, "error");
      deleteBtn.onclick = () => li.remove();
      showToast(`Upload failed (${res.status}): ${res.statusText}`);
      return;
    }
    const { doc_id: docId } = await res.json();
    uploadedFiles.set(docId, { name: file.name, status: "ingesting" });
    deleteBtn.onclick = () => deleteDocument(docId, li, deleteBtn);
    pollDocStatus(docId, badge);
  } catch (err) {
    updateBadge(badge, "error");
    deleteBtn.onclick = () => li.remove();
    showToast("Upload error: " + err.message);
  }
}

function hasReadyDoc() {
  for (const info of uploadedFiles.values()) {
    if (info.status === "ready") return true;
  }
  return false;
}

// Drop zone wiring
dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  for (const file of e.dataTransfer.files) uploadFile(file);
});
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  for (const file of fileInput.files) uploadFile(file);
  fileInput.value = "";
});

// ── Chat ─────────────────────────────────────────────────────────────────────

let isStreaming = false;

function appendUserMessage(text) {
  const li = document.createElement("li");
  li.className = "message user";
  li.textContent = text;
  messages.appendChild(li);
  messages.scrollTop = messages.scrollHeight;
}

function appendAssistantPlaceholder() {
  const li = document.createElement("li");
  li.className = "message assistant";
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  li.appendChild(cursor);
  messages.appendChild(li);
  messages.scrollTop = messages.scrollHeight;
  return li;
}

function finalizeBubble(li, text, isError = false) {
  li.querySelector(".cursor")?.remove();
  if (isError) {
    li.className = "message error";
    li.textContent = text;
  } else {
    li.textContent = text;
  }
  messages.scrollTop = messages.scrollHeight;
}

function enableInput() {
  isStreaming = false;
  sendBtn.disabled = false;
  queryInput.disabled = false;
  queryInput.focus();
}

async function saveTurn(query, answer) {
  const base = { session_id: SESSION_ID, user_id: SESSION_ID };
  const opts = (role, content) => ({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...base, role, content }),
  });
  try {
    await fetch("/memory/turns", opts("user", query));
    await fetch("/memory/turns", opts("assistant", answer));
  } catch {
    // non-critical; do not surface to user
  }
}

async function sendMessage() {
  const query = queryInput.value.trim();
  if (!query || isStreaming) return;

  if (uploadedFiles.size > 0 && !hasReadyDoc()) {
    showToast("Documents still processing — answer may not reflect their content yet.");
  }

  queryInput.value = "";
  queryInput.style.height = "";
  appendUserMessage(query);

  isStreaming = true;
  sendBtn.disabled = true;
  queryInput.disabled = true;

  const bubbleLi = appendAssistantPlaceholder();

  // 1. Submit query to get a job_id
  let queryRes;
  try {
    queryRes = await fetch("/agent/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION_ID, query, top_k: 5 }),
    });
  } catch (err) {
    finalizeBubble(bubbleLi, "Network error: " + err.message, true);
    enableInput();
    return;
  }

  if (!queryRes.ok) {
    const body = await queryRes.json().catch(() => ({}));
    finalizeBubble(bubbleLi, "Error: " + (body.error?.message ?? queryRes.statusText), true);
    enableInput();
    return;
  }

  const { job_id: jobId } = await queryRes.json();

  // 2. Stream the answer via SSE (server sends named events: token / done / error / timeout)
  const es = new EventSource(`/agent/stream/${jobId}?session_id=${SESSION_ID}`);
  let answer = "";
  let finished = false;

  function closeStream() {
    finished = true;
    es.close();
  }

  es.addEventListener("token", (ev) => {
    answer += ev.data;
    bubbleLi.querySelector(".cursor")?.remove();
    bubbleLi.textContent = answer;
    messages.scrollTop = messages.scrollHeight;
  });

  es.addEventListener("done", () => {
    closeStream();
    finalizeBubble(bubbleLi, answer);
    enableInput();
    saveTurn(query, answer);
  });

  es.addEventListener("error", (ev) => {
    closeStream();
    finalizeBubble(bubbleLi, ev.data || "Something went wrong. Please try again.", true);
    enableInput();
  });

  es.addEventListener("timeout", () => {
    closeStream();
    finalizeBubble(bubbleLi, "⏱ Request timed out. Please try again.", true);
    enableInput();
  });

  es.onerror = () => {
    if (finished) return;
    closeStream();
    if (!answer) {
      finalizeBubble(bubbleLi, "Connection error. Please try again.", true);
    } else {
      bubbleLi.querySelector(".cursor")?.remove();
    }
    enableInput();
  };
}

// ── Input keyboard handling ───────────────────────────────────────────────────

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
});

sendBtn.addEventListener("click", sendMessage);
