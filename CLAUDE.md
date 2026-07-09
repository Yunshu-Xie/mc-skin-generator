# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Minecraft skin generator: upload an image → Gemini AI analyzes it → generates a 64×64 Minecraft skin PNG → 3D preview + download. FastAPI backend, vanilla HTML/JS frontend. Each generation request can pick `gemini-2.5-flash` or `gemini-2.5-flash-lite` (both free-tier) via the `ai_model` field, for A/B testing quality.

## Common Commands

```bash
# Install dependencies
pip install fastapi uvicorn openai Pillow python-multipart pydantic-settings pytest httpx ruff

# Run the development server
uvicorn app.main:app --reload

# Run all tests
pytest

# Run a single test file / specific test
pytest tests/test_skin_map.py -v
pytest tests/test_skin_assembler.py::test_assemble_skin_slim -v

# Linting
ruff check .
ruff format .
```

## Configuration

`.env` 文件中需要配置：
- `GEMINI_API_KEY` — Google AI Studio 的 API Key（https://aistudio.google.com/apikey）
- `GEMINI_BASE_URL` — Gemini 的 OpenAI 兼容端点（默认 `https://generativelanguage.googleapis.com/v1beta/openai/`）
- `GEMINI_MODEL_FLASH` / `GEMINI_MODEL_FLASH_LITE` — 两个可选模型的实际模型名（默认 `gemini-2.5-flash` / `gemini-2.5-flash-lite`），由请求里的 `ai_model` 字段（`"flash"` / `"flash-lite"`）选择用哪个
- `GEMINI_DEFAULT_MODEL` — 请求未指定 `ai_model` 时用哪个（默认 `flash`）
- `MAX_IMAGE_DIMENSION` — 上传图片发给 Vision 前先压缩到的最长边像素（默认 `768`），用来控制图片 token 消耗

`app/config.py` 通过 pydantic-settings 读取 `.env`。

## Architecture

### AI Pipeline (`app/services/claude_vision.py`)
通过 OpenAI 兼容 SDK 调用 Gemini API（`GEMINI_BASE_URL`，标准 Bearer token 认证），**单次调用**设计：
- 上传图片先用 Pillow 压缩到 `MAX_IMAGE_DIMENSION` 并重新编码为 JPEG（`_prepare_image`），再 base64 发送
- 一次 Vision 调用同时完成"看图分析"和"画像素"：返回一个 ≤8 色的小调色板 + head 6 个面和 body_front 的调色板索引网格（共 480 个像素，AI 逐像素生成的只有这部分）+ 约 10 个顶层颜色字段（`skin_tone` / `hair_color` / `eye_color` / `shirt_main` / `shirt_shadow` / `arm_main` / `arm_shadow` / `pants_main` / `pants_shadow` / `shoe_color`）
- `ai_model`（`"flash"` / `"flash-lite"`）通过 `_resolve_model_name` 映射到 `GEMINI_MODEL_FLASH` / `GEMINI_MODEL_FLASH_LITE`，逐请求可选，方便对比两个模型的效果，返回的 `metadata.ai_model` 会回显实际用的是哪个
- `_decode_indexed_grid` 把索引网格还原成 hex 颜色网格
- 响应不完整（调色板缺失、网格尺寸不对、必需颜色字段缺失）时整体重试，最多 2 次

### Procedural Fill (`app/services/procedural.py`)
除了 head 和 body_front，其余 29 个基础区域（body 的其余 5 面、双臂全部 6 面、双腿全部 6 面）**不经过 AI**，由 `generate_procedural_regions` 用上面的顶层颜色字段程序化生成：`flat_fill_with_border` 对每个面做纯色填充 + 1px 边框加深阴影。左右两侧直接复用同一套颜色（不需要真正的像素镜像，因为填充规则本身就是对称的）。这是 token 优化的核心：AI 逐像素生成的区域从 ~4096 降到 480，配合合并为单次调用，单次运行 token 消耗从 ~13k-16k 降到 ~2k-3.5k 量级。

### Skin UV Map (`app/services/skin_map.py`)
所有 Minecraft 64×64 皮肤 UV 坐标，以 `FaceRect(x, y, w, h)` dataclass 表示。`PIXEL_KEY_MAP` 将模型输出的 key（如 `"head_front"`）映射到区域组 + 面名。`get_all_regions(model)` 根据 Classic/Slim 返回正确坐标集。

### Skin Assembler (`app/services/skin_assembler.py`)
接收像素数据 dict（AI 解码结果 + procedural 结果合并后的完整 hex 颜色网格）→ 用 Pillow 创建 64×64 RGBA 图片，在正确的 UV 坐标处绘制每个面。验证网格尺寸，静默跳过格式错误的数据。这一层的接口没有变化，仍然只认 `{key: 2D hex 数组}`。

### API Layer (`app/routers/skin.py`)
- `POST /api/generate` — 接收 multipart 图片上传 + model 类型 + 可选 `ai_model`（`"flash"`/`"flash-lite"`，缺省用 `GEMINI_DEFAULT_MODEL`），返回 skin_id + skin_url
- `GET /api/skin/{id}.png` — 从 `skins/` 目录提供生成的 PNG

### Frontend (`app/static/`)
纯 HTML/CSS/JS，无构建步骤。使用 skinview3d@3.1.0（CDN）做 3D 皮肤预览（行走动画）。拖拽上传 + Classic/Slim 切换。

## Key Design Decisions

- **OpenAI 兼容 SDK**: 使用 `openai` Python 包，通过 `base_url` 指向 Gemini 的 OpenAI 兼容端点，方便将来切换其他兼容服务
- **逐请求切换模型**: `ai_model=flash` / `flash-lite` 让两个免费额度模型可以不重启服务直接对比效果
- **Python 3.9 compatibility**: 所有文件使用 `from __future__ import annotations`
- **No database**: 生成的皮肤以 UUID 文件名保存在 `skins/` 目录
- **绝大部分程序化生成**: 只有 head + body_front 由 AI 逐像素生成，其余全部由 `procedural.py` 根据顶层颜色字段规则填充，这是 token 消耗的主要优化点
- **Retry logic**: 单次调用响应不完整时，整体重试最多 2 次（不再是"重试像素生成再合并"，因为现在只有一次调用）
- **Classic vs Slim**: 唯一区别是手臂 front/back 宽度 4→3px，由 `skin_map.py` 坐标选择处理
