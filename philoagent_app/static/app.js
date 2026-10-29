"use strict";

const byId = (id) => document.getElementById(id);
const elements = Object.fromEntries(["workspace", "summary", "state", "provider", "system-name", "system-text", "system-hash", "history", "task-count", "result-content", "result-title", "result-state", "result-error"].map((id) => [id, byId(id)]));
let workspace = "default";
let information = null;
let selected = null;
let activeTab = "answer";
let pending = false;
let polling = null;
let infoSequence = 0;
const statusNames = { completed: "已完成", complete: "已完成", incomplete: "未完整完成", success: "已完成", failed: "失败", error: "失败", interrupted: "已中断", running: "执行中", partial: "部分完成", answer_saved: "作品已保存", scoring: "独立评分中", learning: "经验学习中" };
const text = (value) => value === null || value === undefined ? "—" : typeof value === "string" ? value : JSON.stringify(value, null, 2);
const labelStatus = (value) => statusNames[value] || value || "—";

async function readUtf8(file) {
  try { return new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(await file.arrayBuffer()); }
  catch { throw new Error("请选择 UTF-8 编码的文本文件"); }
}

async function api(path, body) {
  const response = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : { cache: "no-store" });
  const result = await response.json();
  if (!response.ok) throw new Error(result.message || result.error || `请求失败 (${response.status})`);
  return result;
}

function displayResult() {
  const record = selected?.record || selected || {};
  const raw = record.record || record;
  const status = selected?.status || record.status;
  elements["result-state"].textContent = labelStatus(status) === "—" ? "等待任务" : labelStatus(status);
  elements["result-title"].textContent = selected?.operation === "evolve" ? "演化结果" : "作品与过程";
  let value;
  if (activeTab === "answer") {
    value = raw.answer ?? raw.artifact ?? record.answer;
    if (!value && selected?.operation === "evolve") value = selected.result || selected.record || selected;
    if (!value) value = selected ? "作品尚未保存。任务记录和已产生的过程会保留。" : "任务完成后，作品会先保存，再进入独立评分和经验学习。";
  } else if (activeTab === "grade") {
    value = raw.grade ?? record.grade;
    if (value === undefined || value === null) value = "暂无评分。评分缺失或失败不会记为零分。";
    else if (raw.quality_score !== undefined && typeof value === "object") value = { quality_score: raw.quality_score, grade: value };
  } else if (activeTab === "tools") {
    value = raw.agent_trajectory ?? raw.tool_calls ?? record.agent_trajectory ?? "暂无工具记录。";
  } else if (activeTab === "learning") {
    const learned = raw.memory_ingestion_complete ?? record.memory_ingestion_complete;
    value = learned === true ? "本次任务的经验已保存，后续任务会继续使用。" : learned === false ? "经验摄入未完成。已生成的作品保留，不会静默重试。" : "任务完成并评分后，将摄入本次经验。";
    if (raw.memory_ingestion_error || raw.learning_error) value += "\n\n" + text(raw.memory_ingestion_error || raw.learning_error);
  } else {
    value = selected?.api_usage ?? raw.api_usage ?? record.api_usage ?? "暂无实际用量记录。";
  }
  elements["result-content"].textContent = text(value);
  elements["result-content"].classList.toggle("empty", !selected);
  elements["result-content"].setAttribute("aria-labelledby", `tab-${activeTab}`);
  const error = selected?.error || selected?.message || raw.error || raw.error_type;
  elements["result-error"].textContent = error ? `执行记录：${text(error)}` : "";
  document.querySelectorAll("[data-tab]").forEach((button) => { const active = button.dataset.tab === activeTab; button.setAttribute("aria-selected", String(active)); button.tabIndex = active ? 0 : -1; });
}

function setControls() {
  const locked = pending || Boolean(information?.busy);
  byId("run-button").disabled = !information?.initialized || locked;
  byId("evolve-button").disabled = !information?.initialized || locked;
  byId("import-button").disabled = locked;
  elements.workspace.disabled = pending;
  byId("run-note").textContent = !information?.initialized ? "请先导入初始哲学系统。" : locked ? "当前操作结束前，不能开始新任务或演化。" : "将使用已配置的模型；作品与经验会保存到此智能体。";
}

function renderInfo(info) {
  information = info;
  const running = pending || Boolean(info.busy);
  elements.state.textContent = running ? "执行中" : info.initialized ? "已就绪" : "未初始化";
  elements.state.dataset.running = String(running);
  elements.summary.textContent = info.initialized ? `当前智能体 ${workspace} · 延续同一份哲学系统与任务经验` : "导入一份哲学系统，开始第一个任务。";
  elements.provider.textContent = typeof info.provider === "string" ? info.provider : "—";
  const system = info.system || {};
  elements["system-name"].textContent = system.filename || "尚未导入";
  elements["system-text"].textContent = Array.isArray(system.items) ? system.items.join("\n\n") : system.content || "导入后会在这里显示完整内容。";
  elements["system-hash"].textContent = system.sha256 ? `来源 SHA-256\n${system.sha256}` : "";
  const tasks = Array.isArray(info.tasks) ? info.tasks : [];
  elements["task-count"].textContent = `${tasks.length} 项`;
  const previousId = selected?.operation_id || selected?.id;
  elements.history.replaceChildren();
  if (!tasks.length) {
    const empty = document.createElement("p"); empty.className = "empty"; empty.textContent = "完成的作品与未完成的任务会留在这里。"; elements.history.append(empty);
  }
  for (const task of tasks) {
    const button = document.createElement("button"); button.type = "button"; button.className = "history-item";
    button.setAttribute("aria-current", String((task.operation_id || task.id) === previousId));
    const question = document.createElement("span"); question.textContent = task.question || task.record?.question || task.task_id || "任务";
    const status = document.createElement("span"); status.textContent = `${labelStatus(task.status)}${task.quality_score === null || task.quality_score === undefined ? "" : ` · ${Number(task.quality_score).toFixed(1)} / 100`}`;
    button.append(question, status); button.addEventListener("click", () => { selected = task; displayResult(); renderInfo(information); }); elements.history.append(button);
  }
  const latest = info.last_operation;
  if (latest && (running || !selected || latest.operation_id === previousId)) selected = latest.result || latest;
  else if (previousId) selected = tasks.find((task) => (task.operation_id || task.id) === previousId) || selected;
  else if (tasks.length) selected = tasks[0];
  setControls(); displayResult();
}

async function refreshInfo() {
  const sequence = ++infoSequence;
  const name = workspace;
  try {
    const info = await api(`/api/info?workspace=${encodeURIComponent(name)}`);
    if (sequence !== infoSequence || name !== workspace) return;
    renderInfo(info); byId("connection-message").textContent = "";
    if (!pending && !info.busy && polling) { clearInterval(polling); polling = null; }
  } catch (error) {
    if (sequence !== infoSequence || name !== workspace) return;
    byId("connection-message").textContent = `状态读取失败：${error.message}。刷新只读取记录，不会重跑任务。`;
  }
}

async function loadWorkspaces(chosen) {
  const config = await api("/api/config");
  workspace = chosen || config.initial_workspace;
  const names = [...new Set([workspace, ...config.workspaces])];
  elements.workspace.replaceChildren(...names.map((name) => { const option = document.createElement("option"); option.value = name; option.textContent = name; return option; }));
  elements.workspace.value = workspace;
  await refreshInfo();
}

function requestId() { return typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`; }

async function perform(path, body, errorId) {
  if (pending || information?.busy) return;
  pending = true; byId(errorId).textContent = ""; setControls(); elements.state.textContent = "执行中"; elements.state.dataset.running = "true";
  polling = setInterval(refreshInfo, 1800);
  try {
    const result = await api(path, { ...body, workspace, request_id: requestId() });
    selected = result; activeTab = "answer"; displayResult();
  } catch (error) {
    byId(errorId).textContent = `${error.message}。已保存记录仍可查看；不会自动重试。`;
  } finally {
    clearInterval(polling); polling = null; pending = false; await refreshInfo();
  }
}

byId("import-form").addEventListener("submit", async (event) => {
  event.preventDefault(); if (pending || information?.busy) return;
  const file = byId("system-file").files[0]; if (!file) return;
  byId("import-error").textContent = "";
  if (file.size > 1024 * 1024) { byId("import-error").textContent = "系统文件不得超过 1 MiB。"; return; }
  const name = byId("agent-name").value.trim();
  pending = true; setControls();
  try {
    await api("/api/init", { name, filename: file.name, content: await readUtf8(file) });
    selected = null; await loadWorkspaces(name); byId("import-details").open = false;
    byId("system-details").open = true;
  } catch (error) { byId("import-error").textContent = error.message; }
  finally { pending = false; await refreshInfo(); }
});

byId("task-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const materials = byId("materials").value.trim();
  const task = { domain: byId("domain").value, question: byId("question").value.trim(), materials: materials ? [materials] : [] };
  const deliverable = byId("deliverable").value.trim(); if (deliverable) task.deliverable = deliverable;
  if (!task.question) { byId("run-error").textContent = "请填写任务问题。"; return; }
  selected = null; await perform("/api/run", { task }, "run-error");
});

byId("evolve-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = byId("dev-file").files[0]; if (!file) return;
  byId("evolve-error").textContent = "";
  if (file.size > 1024 * 1024) { byId("evolve-error").textContent = "开发题集不得超过 1 MiB。"; return; }
  let tasks;
  try { tasks = (await readUtf8(file)).replace(/^\uFEFF/, "").split(/\r?\n/).filter((line) => line.trim()).map((line, index) => { try { return JSON.parse(line); } catch { throw new Error(`第 ${index + 1} 行不是合法 JSON`); } }); }
  catch (error) { byId("evolve-error").textContent = error.message; return; }
  if (tasks.length < 4) { byId("evolve-error").textContent = "手动演化至少需要四道独立开发题。"; return; }
  await perform("/api/evolve", { tasks, rounds: Number(byId("rounds").value) }, "evolve-error");
});

const tabs = [...document.querySelectorAll("[data-tab]")];
tabs.forEach((button, index) => {
  button.addEventListener("click", () => { activeTab = button.dataset.tab; displayResult(); });
  button.addEventListener("keydown", (event) => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else return;
    event.preventDefault(); activeTab = tabs[next].dataset.tab; displayResult(); tabs[next].focus();
  });
});
elements.workspace.addEventListener("change", async () => { workspace = elements.workspace.value; selected = null; await refreshInfo(); });
byId("refresh").addEventListener("click", refreshInfo);
loadWorkspaces().then(() => { if (information?.initialized) byId("import-details").open = false; if (information?.busy) polling = setInterval(refreshInfo, 1800); }).catch((error) => { byId("connection-message").textContent = `无法连接本机服务：${error.message}`; });
