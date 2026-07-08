/**
 * ComfyUI-CoreAI browser extension.
 *
 * Adds a status badge and download button to every CoreAI node so the
 * user can see model install state at a glance and trigger downloads
 * without leaving the canvas.
 *
 * Also adds a "coreai_perf" badge widget that surfaces inference timing
 * (tok/s, ms, compute unit) returned by the runner in the `ui` hidden
 * output. The badge renders green "N cached" when prefix-cache reuse
 * fires, matching the zoo ChatView stats bar pattern.
 *
 * Talks to the ComfyUI server via /coreai/* routes (proxied to the runner).
 */

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const PLUGIN_NAME = "ComfyUI-CoreAI";
const COREAI_NODE_PREFIX = "CoreAI";

// Nodes that don't use the standard model dropdown (no download UI).
// Perf badge is only relevant on inference nodes — utility nodes are skipped.
const UTILITY_NODES = new Set(["CoreAIHealthCheck", "CoreAIAppleText"]);

/**
 * Add a read-only status widget to a CoreAI node.
 */
function addStatusWidget(node) {
    if (UTILITY_NODES.has(node.constructor.nodeData?.name)) return;

    const widget = node.addWidget(
        "text",
        "coreai_status",
        "Checking...",
        () => {},
        { serialize: false }
    );
    widget.disabled = true;

    // Refresh when the model widget changes
    const origOnPropertyChanged = node.onPropertyChanged;
    node.onPropertyChanged = function (property) {
        if (property === "model") refreshStatus(node);
        if (origOnPropertyChanged) origOnPropertyChanged.call(this, property);
    };
}

/**
 * Add a download button widget to a CoreAI node.
 */
function addDownloadButton(node) {
    if (UTILITY_NODES.has(node.constructor.nodeData?.name)) return;

    let btn = node.addWidget("button", "coreai_download", "Download", async () => {
        const modelWidget = node.widgets?.find((w) => w.name === "model");
        if (!modelWidget) return;

        btn.name = "Starting...";
        try {
            await api.fetchApi(
                `/coreai/models/${modelWidget.value}/download`,
                { method: "POST" }
            );
            btn.name = "Downloading...";
            pollProgress(node, modelWidget.value, btn);
        } catch (e) {
            btn.name = "Download Failed";
            setTimeout(() => { btn.name = "Download"; }, 3000);
        }
    });
}

/**
 * Build the DOM element for the perf badge widget.
 * Renders a monospaced stats line + an optional green "N cached" badge.
 */
function buildPerfElement(node) {
    const wrap = document.createElement("div");
    wrap.className = "coreai-perf";
    Object.assign(wrap.style, {
        display: "flex",
        alignItems: "center",
        gap: "6px",
        padding: "2px 4px",
        minHeight: "18px",
        fontFamily: "var(--font-mono, monospace)",
        fontFeatureSettings: '"tnum" 1',  // monospaced digits — no jitter
        fontSize: "11px",
        opacity: "0.85",
    });

    const stats = document.createElement("span");
    stats.className = "coreai-perf-stats";
    stats.textContent = "—";
    // tok/s value is the headline metric — render it semibold (zoo pattern)
    stats.style.fontWeight = "500";
    wrap.appendChild(stats);

    const cached = document.createElement("span");
    cached.className = "coreai-perf-cached";
    cached.hidden = true;
    Object.assign(cached.style, {
        background: "#1f7a3d",
        color: "#fff",
        borderRadius: "8px",
        padding: "1px 6px",
        fontSize: "10px",
        fontWeight: "600",
        whiteSpace: "nowrap",
    });
    wrap.appendChild(cached);

    return wrap;
}

/**
 * Apply a perf payload (from the node's `ui.coreai_perf` output) to the badge.
 */
function applyPerf(node, perf) {
    const widget = node.widgets?.find((w) => w.name === "coreai_perf");
    const el = widget?.element;
    if (!el || !perf) return;

    const stats = el.querySelector(".coreai-perf-stats");
    const cached = el.querySelector(".coreai-perf-cached");

    const text = perf.text || "—";

    // For chat/VLM nodes, bold the leading "N tok/s" headline (zoo pattern).
    // The remainder (ms, unit) stays at the container's normal weight.
    const tps = perf.tokens_per_second;
    const tpsStr = tps != null ? `${tps} tok/s` : null;
    stats.textContent = "";
    if (tpsStr && text.startsWith(tpsStr)) {
        const head = document.createElement("b");
        head.textContent = tpsStr;
        head.style.fontWeight = "700";
        stats.appendChild(head);
        stats.appendChild(document.createTextNode(text.slice(tpsStr.length)));
    } else {
        stats.textContent = text;
    }

    // Green "N cached" badge when prefix reuse fired.
    const reused = perf.reused_prompt_tokens || 0;
    if (cached) {
        if (reused > 0) {
            cached.textContent = `${reused} cached`;
            cached.hidden = false;
        } else {
            cached.hidden = true;
        }
    }
}

/**
 * Add a perf badge DOM widget to a CoreAI node.
 */
function addPerfWidget(node) {
    if (UTILITY_NODES.has(node.constructor.nodeData?.name)) return;

    const element = buildPerfElement(node);
    // addDOMWidget renders the element inside the node body and handles
    // layout/serialization. serialize:false keeps it out of the saved graph.
    const widget = node.addDOMWidget?.(
        "coreai_perf",
        "coreai_perf",
        element,
        { serialize: false }
    );
    if (widget) {
        widget.element = element;
        widget.computeSize = () => [200, 24];
    }
}

/**
 * Fetch and display the current status of the selected model.
 */
async function refreshStatus(node) {
    const modelWidget = node.widgets?.find((w) => w.name === "model");
    if (!modelWidget) return;

    const statusWidget = node.widgets?.find((w) => w.name === "coreai_status");
    if (!statusWidget) return;

    try {
        // Fetch status + catalog metadata in parallel
        const [statusResp, catalogResp] = await Promise.all([
            api.fetchApi(`/coreai/models/${modelWidget.value}/status`),
            api.fetchApi(`/coreai/catalog/model/${modelWidget.value}`),
        ]);

        const status = await statusResp.json();
        const catalogInfo = await catalogResp.json();
        const sizeStr = catalogInfo?.size?.artifact_size || "";
        const precision = catalogInfo?.size?.precision || "";

        if (status.loaded) {
            statusWidget.value = `✓ Loaded${sizeStr ? " · " + sizeStr : ""}`;
        } else if (status.installed) {
            statusWidget.value = `✓ Ready${sizeStr ? " · " + sizeStr : ""}`;
        } else if (status.download) {
            const pct = Math.round(status.download.fraction * 100);
            statusWidget.value = `↓ ${pct}%${sizeStr ? " of " + sizeStr : ""}`;
        } else {
            const meta = [sizeStr, precision].filter(Boolean).join(" · ");
            statusWidget.value = `Not installed${meta ? " · " + meta : ""}`;
        }

        // Update download button label
        const btn = node.widgets?.find((w) => w.name === "coreai_download");
        if (btn) {
            if (status.loaded || status.installed) {
                btn.name = "Installed ✓";
            } else if (status.download) {
                btn.name = `Downloading ${Math.round(status.download.fraction * 100)}%`;
            } else {
                btn.name = "Download";
            }
        }
    } catch {
        // Runner not started yet — normal before first predict
        statusWidget.value = "Runner idle — run workflow to start";
    }
}

/**
 * Poll download progress until complete.
 */
async function pollProgress(node, modelId, btn) {
    const interval = setInterval(async () => {
        try {
            const resp = await api.fetchApi(`/coreai/models/${modelId}/status`);
            const data = await resp.json();

            if (data.loaded || data.installed) {
                clearInterval(interval);
                btn.name = "Installed ✓";
                refreshStatus(node);
            } else if (data.download) {
                const pct = Math.round(data.download.fraction * 100);
                btn.name = `Downloading ${pct}%`;

                // Update status widget too
                const statusWidget = node.widgets?.find(
                    (w) => w.name === "coreai_status"
                );
                if (statusWidget) {
                    statusWidget.value = `↓ ${pct}%`;
                }
            }
        } catch {
            // keep polling
        }
    }, 1000);

    // Stop polling after 10 minutes (safety)
    setTimeout(() => clearInterval(interval), 600000);
}

app.registerExtension({
    name: PLUGIN_NAME,
    async nodeCreated(node) {
        if (!node.constructor.nodeData?.name?.startsWith(COREAI_NODE_PREFIX)) return;

        addStatusWidget(node);
        addDownloadButton(node);
        addPerfWidget(node);

        // Initial status fetch after a short delay (node needs to be fully rendered)
        setTimeout(() => refreshStatus(node), 500);
    },
});

// --- Perf badge updates -----------------------------------------------------
//
// Listen for the ComfyUI "executed" message. Each CoreAI node returns its
// perf data as a hidden `ui.coreai_perf` output; the message carries
// `detail.node` (the node id) and `detail.output.coreai_perf` (our payload).
// We map the id back to the graph node and update its badge.
api.addEventListener("executed", (e) => {
    const detail = e?.detail;
    if (!detail) return;
    const perf = detail.output?.coreai_perf;
    if (!perf) return;

    const graphNode = app.graph?.getNodeById?.(detail.node);
    if (!graphNode) return;

    const name = graphNode.constructor?.nodeData?.name;
    if (!name || !name.startsWith(COREAI_NODE_PREFIX)) return;

    applyPerf(graphNode, perf);
});
