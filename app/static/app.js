// ── DOM Elements ──
const uploadForm = document.getElementById("uploadForm");
const dropZone = document.getElementById("dropZone");
const dropPrompt = document.getElementById("dropPrompt");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const generateBtn = document.getElementById("generateBtn");
const statusDiv = document.getElementById("status");
const statusText = document.getElementById("statusText");
const errorDiv = document.getElementById("error");
const viewerSection = document.getElementById("viewerSection");
const skinDescription = document.getElementById("skinDescription");
const downloadBtn = document.getElementById("downloadBtn");
const resetBtn = document.getElementById("resetBtn");

let viewer = null;
let currentSkinUrl = null;

// ── Drop Zone ──
dropZone.addEventListener("click", () => imageInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) {
        imageInput.files = e.dataTransfer.files;
        showPreview(e.dataTransfer.files[0]);
    }
});

imageInput.addEventListener("change", () => {
    if (imageInput.files.length) {
        showPreview(imageInput.files[0]);
    }
});

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.hidden = false;
        dropPrompt.hidden = true;
        generateBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// ── Form Submit ──
uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!imageInput.files.length) return;

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);
    formData.append("model", document.querySelector('input[name="model"]:checked').value);
    formData.append("style_notes", document.getElementById("styleNotes").value);
    formData.append("ai_model", document.querySelector('input[name="aiModel"]:checked').value);

    // Show loading
    generateBtn.disabled = true;
    errorDiv.hidden = true;
    viewerSection.hidden = true;
    statusDiv.hidden = false;
    statusText.textContent = "分析图片中...";

    try {
        // Update status as time passes
        const statusTimer = setTimeout(() => {
            statusText.textContent = "设计皮肤中...";
        }, 4000);

        const statusTimer2 = setTimeout(() => {
            statusText.textContent = "拼装纹理中...";
        }, 12000);

        const response = await fetch("/api/generate", {
            method: "POST",
            body: formData,
        });

        clearTimeout(statusTimer);
        clearTimeout(statusTimer2);

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "生成失败");
        }

        statusDiv.hidden = true;
        showViewer(data.skin_url, data.model, data);
    } catch (err) {
        statusDiv.hidden = true;
        errorDiv.textContent = `❌ ${err.message}`;
        errorDiv.hidden = false;
        generateBtn.disabled = false;
    }
});

// ── 3D Viewer ──
function showViewer(skinUrl, model, result) {
    viewerSection.hidden = false;
    currentSkinUrl = skinUrl;

    const metadata = (result && result.metadata) || {};
    if (metadata.description) {
        const modelLabel = metadata.ai_model === "flash-lite" ? "Flash-Lite" : "Flash";
        const fallback = metadata.layout_source === "fallback" ? " · 默认版式" : "";
        skinDescription.textContent = `[${modelLabel}${fallback}] ${metadata.description}`;
    }

    showPalette(result);
    showFidelity(result);

    if (viewer) {
        viewer.dispose();
    }

    const canvas = document.getElementById("skinViewer");
    const width = Math.min(400, window.innerWidth - 60);

    viewer = new skinview3d.SkinViewer({
        canvas: canvas,
        width: width,
        height: Math.round(width * 1.4),
        skin: skinUrl,
        model: model === "slim" ? "slim" : "default",
    });

    viewer.autoRotate = true;
    viewer.autoRotateSpeed = 0.5;
    viewer.animation = new skinview3d.WalkingAnimation();
    viewer.animation.speed = 0.6;
    viewer.zoom = 0.9;
}

// ── Download ──
downloadBtn.addEventListener("click", () => {
    if (!currentSkinUrl) return;
    const a = document.createElement("a");
    a.href = currentSkinUrl;
    a.download = "minecraft_skin.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
});

// ── Reset ──
resetBtn.addEventListener("click", () => {
    viewerSection.hidden = true;
    if (viewer) {
        viewer.dispose();
        viewer = null;
    }
    currentSkinUrl = null;
    preview.hidden = true;
    preview.src = "";
    dropPrompt.hidden = false;
    imageInput.value = "";
    generateBtn.disabled = true;
    errorDiv.hidden = true;
    document.getElementById("styleNotes").value = "";
});


// ── Palette & fidelity ──
// The whole skin is drawn from one palette; the first slots are semantic
// roles, so a viewer can see at a glance which color is "the shirt".
function showPalette(result) {
    const strip = document.getElementById("paletteStrip");
    const palette = (result && result.palette) || [];
    strip.innerHTML = "";
    strip.hidden = palette.length === 0;

    const roleOf = {};
    Object.entries((result && result.roles) || {}).forEach(([role, i]) => {
        roleOf[i] = role;
    });

    palette.forEach((hex, i) => {
        const swatch = document.createElement("span");
        swatch.className = roleOf[i] ? "swatch role" : "swatch";
        swatch.style.background = hex;
        swatch.title = roleOf[i] ? `${roleOf[i]} · ${hex}` : hex;
        strip.appendChild(swatch);
    });
}

// SSIM and detail are both area-weighted, so they are shown together and
// neither is called a score — see docs/ARCHITECTURE.md section 7.
function showFidelity(result) {
    const line = document.getElementById("fidelity");
    const head = result && result.metrics && result.metrics.head_front;
    if (!head) {
        line.hidden = true;
        return;
    }
    line.textContent =
        `面部 SSIM ${head.ssim.toFixed(3)} · ΔE ${head.delta_e_mean.toFixed(3)} ` +
        `· 细节保留 ${head.detail.toFixed(2)}`;
    line.hidden = false;
}
