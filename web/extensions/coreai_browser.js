/**
 * ComfyUI-CoreAI browser extension.
 *
 * Adds a status badge and download button to every CoreAI node so the
 * user can see model install state at a glance and trigger downloads
 * without leaving the canvas.
 *
 * Talks to the ComfyUI server via /coreai/* routes (proxied to the runner).
 */

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const PLUGIN_NAME = "ComfyUI-CoreAI";
const COREAI_NODE_PREFIX = "CoreAI";

// Nodes that don't use the standard model dropdown (no download UI)
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

        // Initial status fetch after a short delay (node needs to be fully rendered)
        setTimeout(() => refreshStatus(node), 500);
    },
});
