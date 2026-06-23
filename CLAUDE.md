# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Minecraft skin generator: upload an image → CodeBuddy AI (腾讯) analyzes it → generates a 64×64 Minecraft skin PNG → 3D preview + download. FastAPI backend, vanilla HTML/JS frontend.

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
- `CODEBUDDY_API_KEY` — CodeBuddy Coding Plan 的 API Key（格式 `sk-sp-xxxx`）
- `CODEBUDDY_BASE_URL` — API 端点（默认 `https://api.lkeap.cloud.tencent.com/coding/v3`）
- `CODEBUDDY_VISION_MODEL` — 用于图片分析的模型名（默认 `hunyuan-2.0-instruct`）
- `CODEBUDDY_TEXT_MODEL` — 用于像素生成的模型名（默认 `hunyuan-2.0-instruct`）

`app/config.py` 通过 pydantic-settings 读取 `.env`。

## Architecture

### AI Pipeline (`app/services/claude_vision.py`)
通过 OpenAI 兼容 SDK 调用 CodeBuddy API，两步设计：
1. **Image Analysis** — Vision 模型分析上传照片，输出 JSON（每个身体部位的颜色调色板和描述）
2. **Pixel Generation** — 第二次调用（纯文本，不发图片），根据分析 JSON 输出每个面的 2D hex 颜色数组

这个拆分是故意的：单次输出全部 ~4096 像素不可靠。两步分离"理解图片"和"画像素画"。

### Skin UV Map (`app/services/skin_map.py`)
所有 Minecraft 64×64 皮肤 UV 坐标，以 `FaceRect(x, y, w, h)` dataclass 表示。`PIXEL_KEY_MAP` 将模型输出的 key（如 `"head_front"`）映射到区域组 + 面名。`get_all_regions(model)` 根据 Classic/Slim 返回正确坐标集。

### Skin Assembler (`app/services/skin_assembler.py`)
接收像素数据 dict → 用 Pillow 创建 64×64 RGBA 图片，在正确的 UV 坐标处绘制每个面。验证网格尺寸，静默跳过格式错误的数据。

### API Layer (`app/routers/skin.py`)
- `POST /api/generate` — 接收 multipart 图片上传 + model 类型，返回 skin_id + skin_url
- `GET /api/skin/{id}.png` — 从 `skins/` 目录提供生成的 PNG

### Frontend (`app/static/`)
纯 HTML/CSS/JS，无构建步骤。使用 skinview3d@3.1.0（CDN）做 3D 皮肤预览（行走动画）。拖拽上传 + Classic/Slim 切换。

## Key Design Decisions

- **OpenAI 兼容 SDK**: 使用 `openai` Python 包，通过 `base_url` 指向 CodeBuddy 端点，方便将来切换其他兼容服务
- **Python 3.9 compatibility**: 所有文件使用 `from __future__ import annotations`
- **No database**: 生成的皮肤以 UUID 文件名保存在 `skins/` 目录
- **Retry logic**: 像素数据不完整（<36 个基础区域）时，自动重试最多 2 次并合并结果
- **Classic vs Slim**: 唯一区别是手臂 front/back 宽度 4→3px，由 `skin_map.py` 坐标选择处理
