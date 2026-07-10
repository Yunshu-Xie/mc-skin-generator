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
let currentSkinId = null;

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
        showViewer(data.skin_id, data.skin_url, data.model, data.metadata);
    } catch (err) {
        statusDiv.hidden = true;
        errorDiv.textContent = `❌ ${err.message}`;
        errorDiv.hidden = false;
        generateBtn.disabled = false;
    }
});

// ── 3D Viewer ──
function showViewer(skinId, skinUrl, model, metadata) {
    viewerSection.hidden = false;
    currentSkinId = skinId;
    currentSkinUrl = skinUrl;

    if (metadata && metadata.description) {
        const modelLabel = metadata.ai_model === "flash-lite" ? "Flash-Lite" : "Flash";
        skinDescription.textContent = `[${modelLabel}] ${metadata.description}`;
    }

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
    currentSkinId = null;
    preview.hidden = true;
    preview.src = "";
    dropPrompt.hidden = false;
    imageInput.value = "";
    generateBtn.disabled = true;
    errorDiv.hidden = true;
    document.getElementById("styleNotes").value = "";
});
