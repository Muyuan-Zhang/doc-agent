// Session identity — persisted across page reloads
const SESSION_ID = (() => {
  const stored = localStorage.getItem("session_id");
  if (stored) return stored;
  const id = crypto.randomUUID();
  localStorage.setItem("session_id", id);
  return id;
})();

// ── Panel switching ──────────────────────────────────────────────────────────
function switchPanel(panelName) {
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));

  document.getElementById(`${panelName}-panel`)?.classList.add("active");
  document.querySelector(`[data-panel="${panelName}"]`)?.classList.add("active");
  document.getElementById("app")?.classList.toggle("tool-mode", panelName !== "chat");
}

document.querySelectorAll(".nav-tab").forEach(tab => {
  tab.addEventListener("click", () => switchPanel(tab.dataset.panel));
});

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
    if (res.ok) {
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

// ── F5: Retrieval Debug Panel ─────────────────────────────────────────────────

const retrievalQuery = document.getElementById("retrieval-query");
const retrievalTopK = document.getElementById("retrieval-top-k");
const retrievalSearchBtn = document.getElementById("retrieval-search-btn");
const retrievalResults = document.getElementById("retrieval-results");

retrievalSearchBtn.addEventListener("click", async () => {
  const query = retrievalQuery.value.trim();
  const topK = parseInt(retrievalTopK.value) || 5;
  
  if (!query) {
    showToast("Please enter a search query");
    return;
  }
  
  retrievalSearchBtn.disabled = true;
  retrievalResults.innerHTML = "";
  
  try {
    const res = await fetch("/retrieval/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      retrievalResults.innerHTML = `<p style="color: red;">Error: ${err.detail || res.statusText}</p>`;
      showToast("Retrieval failed");
      retrievalSearchBtn.disabled = false;
      return;
    }
    
    const { chunks } = await res.json();
    if (chunks.length === 0) {
      retrievalResults.innerHTML = "<p>No results found</p>";
    } else {
      retrievalResults.innerHTML = chunks.map((chunk, idx) => `
        <div class="chunk-item">
          <div class="chunk-item-header">Chunk ${idx + 1} (doc_id: ${chunk.doc_id})</div>
          <div class="chunk-item-content">${escapeHtml(chunk.content)}</div>
        </div>
      `).join("");
    }
    showToast(`Found ${chunks.length} results`);
  } catch (err) {
    retrievalResults.innerHTML = `<p style="color: red;">Error: ${err.message}</p>`;
    showToast("Retrieval error: " + err.message);
  } finally {
    retrievalSearchBtn.disabled = false;
  }
});

// ── F3: Cache Management Panel ────────────────────────────────────────────────

const cacheRefreshStatsBtn = document.getElementById("cache-refresh-stats");
const cacheLoadReviewBtn = document.getElementById("cache-load-review");
const cacheStats = document.getElementById("cache-stats");
const cacheReview = document.getElementById("cache-review");

async function loadCacheStats() {
  try {
    const res = await fetch("/cache/stats");
    if (!res.ok) throw new Error("Failed to fetch stats");
    
    const { hits, misses, pending } = await res.json();
    const total = hits + misses;
    const rate = total > 0 ? ((hits / total) * 100).toFixed(1) : 0;
    
    cacheStats.innerHTML = `
      <div class="stat-item">
        <div class="stat-value">${hits}</div>
        <div class="stat-label">Hits</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">${misses}</div>
        <div class="stat-label">Misses</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">${rate}%</div>
        <div class="stat-label">Hit Rate</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">${pending}</div>
        <div class="stat-label">Pending</div>
      </div>
    `;
  } catch (err) {
    cacheStats.innerHTML = `<p style="color: red;">Error: ${err.message}</p>`;
    showToast("Failed to load cache stats");
  }
}

async function loadCacheReview() {
  try {
    const res = await fetch("/cache/review");
    if (!res.ok) throw new Error("Failed to fetch reviews");
    
    const { pending } = await res.json();
    if (pending.length === 0) {
      cacheReview.innerHTML = "<p>No pending reviews</p>";
      return;
    }
    
    cacheReview.innerHTML = pending.map(item => `
      <div class="review-item">
        <div class="review-item-header">
          <div class="review-item-query">${escapeHtml(item.original_query)}</div>
        </div>
        <div class="review-item-meta">
          Normalized: ${escapeHtml(item.normalized_query)} | ${item.chunk_count} chunks | Approvals: ${item.approval_count}
        </div>
        <div class="review-item-actions">
          <button class="btn-approve" onclick="approveCacheEntry('${item.query_hash}')">Approve</button>
          <button class="btn-reject" onclick="rejectCacheEntry('${item.query_hash}')">Reject</button>
          <button class="btn-delete" onclick="deleteCacheEntry('${item.query_hash}')">Delete</button>
        </div>
      </div>
    `).join("");
  } catch (err) {
    cacheReview.innerHTML = `<p style="color: red;">Error: ${err.message}</p>`;
    showToast("Failed to load reviews");
  }
}

async function approveCacheEntry(queryHash) {
  const reviewerId = prompt("Enter your reviewer ID:");
  if (!reviewerId) return;
  
  try {
    const headers = { "Content-Type": "application/json" };
    const apiKey = localStorage.getItem("cache_api_key");
    if (apiKey) headers["X-API-Key"] = apiKey;
    
    const res = await fetch(`/cache/review/${queryHash}/approve`, {
      method: "POST",
      headers,
      body: JSON.stringify({ reviewer_id: reviewerId }),
    });
    if (res.ok) {
      showToast("Entry approved");
      loadCacheReview();
    } else {
      const err = await res.json().catch(() => ({}));
      showToast("Approval failed: " + (err.detail || res.statusText));
    }
  } catch (err) {
    showToast("Error: " + err.message);
  }
}

async function rejectCacheEntry(queryHash) {
  try {
    const headers = {};
    const apiKey = localStorage.getItem("cache_api_key");
    if (apiKey) headers["X-API-Key"] = apiKey;
    
    const res = await fetch(`/cache/review/${queryHash}/reject`, {
      method: "POST",
      headers,
    });
    if (res.ok) {
      showToast("Entry rejected");
      loadCacheReview();
    } else {
      showToast("Rejection failed: " + res.statusText);
    }
  } catch (err) {
    showToast("Error: " + err.message);
  }
}

async function deleteCacheEntry(queryHash) {
  if (!confirm("Delete this cache entry?")) return;
  try {
    const headers = {};
    const apiKey = localStorage.getItem("cache_api_key");
    if (apiKey) headers["X-API-Key"] = apiKey;
    
    const res = await fetch(`/cache/${queryHash}`, {
      method: "DELETE",
      headers,
    });
    if (res.ok) {
      showToast("Entry deleted");
      loadCacheReview();
    } else {
      showToast("Delete failed: " + res.statusText);
    }
  } catch (err) {
    showToast("Error: " + err.message);
  }
}

cacheRefreshStatsBtn.addEventListener("click", loadCacheStats);
cacheLoadReviewBtn.addEventListener("click", loadCacheReview);

// ── F4: Memory Management Panel ───────────────────────────────────────────────

const memoryLoadContextBtn = document.getElementById("memory-load-context");
const memorySummarizeBtn = document.getElementById("memory-summarize");
const memoryAddFactBtn = document.getElementById("memory-add-fact");
const memoryNewFact = document.getElementById("memory-new-fact");
const memoryContext = document.getElementById("memory-context");
const memoryFacts = document.getElementById("memory-facts");

async function loadMemoryContext() {
  try {
    const res = await fetch(`/memory/context/${SESSION_ID}?user_id=${SESSION_ID}`);
    if (!res.ok) throw new Error("Failed to fetch context");
    
    const ctx = await res.json();
    let html = `<h4>Recent Turns: ${ctx.recent_turns?.length || 0}</h4>`;
    
    if (ctx.recent_turns?.length > 0) {
      html += "<pre>" + escapeHtml(JSON.stringify(ctx.recent_turns, null, 2)) + "</pre>";
    }
    
    if (ctx.summary) {
      html += `<h4 style="margin-top: 12px;">Summary</h4><pre>${escapeHtml(ctx.summary)}</pre>`;
    }
    
    if (ctx.static_facts?.length > 0) {
      html += `<h4 style="margin-top: 12px;">Static Facts: ${ctx.static_facts.length}</h4>`;
      html += "<pre>" + escapeHtml(JSON.stringify(ctx.static_facts, null, 2)) + "</pre>";
    }
    
    memoryContext.innerHTML = html;
    showToast("Memory context loaded");
  } catch (err) {
    memoryContext.innerHTML = `<p style="color: red;">Error: ${err.message}</p>`;
    showToast("Failed to load memory context");
  }
}

async function summarizeMemory() {
  try {
    const res = await fetch(`/memory/summarize/${SESSION_ID}?user_id=${SESSION_ID}`, {
      method: "POST",
    });
    if (!res.ok) throw new Error("Summarization failed");
    
    const result = await res.json();
    showToast("Memory summarized: " + result.summary?.substring(0, 100));
    loadMemoryContext();
  } catch (err) {
    showToast("Summarization error: " + err.message);
  }
}

async function addStaticFact() {
  const content = memoryNewFact.value.trim();
  if (!content) {
    showToast("Please enter a fact");
    return;
  }
  
  try {
    const res = await fetch("/memory/static", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: SESSION_ID, content }),
    });
    if (!res.ok) throw new Error("Failed to add fact");
    
    memoryNewFact.value = "";
    showToast("Fact added");
    loadStaticFacts();
    loadMemoryContext();
  } catch (err) {
    showToast("Error: " + err.message);
  }
}

async function loadStaticFacts() {
  try {
    const res = await fetch(`/memory/context/${SESSION_ID}?user_id=${SESSION_ID}`);
    if (!res.ok) throw new Error("Failed to fetch facts");
    
    const ctx = await res.json();
    const facts = ctx.static_facts || [];
    
    if (facts.length === 0) {
      memoryFacts.innerHTML = "<p>No static facts yet</p>";
      return;
    }
    
    memoryFacts.innerHTML = facts.map(fact => `
      <div class="fact-item">
        <div class="fact-content">${escapeHtml(fact.content)}</div>
        <button class="fact-delete" onclick="deleteStaticFact('${fact.id}')">Delete</button>
      </div>
    `).join("");
  } catch (err) {
    memoryFacts.innerHTML = `<p style="color: red;">Error: ${err.message}</p>`;
  }
}

async function deleteStaticFact(factId) {
  if (!confirm("Delete this fact?")) return;
  
  try {
    const res = await fetch(`/memory/static/${factId}?user_id=${SESSION_ID}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new Error("Failed to delete fact");
    
    showToast("Fact deleted");
    loadStaticFacts();
    loadMemoryContext();
  } catch (err) {
    showToast("Error: " + err.message);
  }
}

memoryLoadContextBtn.addEventListener("click", loadMemoryContext);
memorySummarizeBtn.addEventListener("click", summarizeMemory);
memoryAddFactBtn.addEventListener("click", addStaticFact);

// ── Utility function ──────────────────────────────────────────────────────────

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
