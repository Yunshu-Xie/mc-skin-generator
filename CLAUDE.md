# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Minecraft skin generator: upload an image → Gemini AI analyzes it → generates a 64×64 Minecraft skin PNG → 3D preview + download. FastAPI backend, vanilla HTML/JS frontend. Each generation request can pick `gemini-2.5-flash` or `gemini-2.5-flash-lite` (both free-tier) via the `ai_model` field, for A/B testing quality.

## Common Commands

```bash
# Install dependencies
pip install fastapi uvicorn openai Pillow python-multipart pydantic-settings pytest pytest-asyncio httpx ruff

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
通过 OpenAI 兼容 SDK 调用 Gemini API（`GEMINI_BASE_URL`，标准 Bearer token 认证），**单次调用**设计，但这次调用不再画任何像素：
- 上传图片先用 Pillow 压缩到 `MAX_IMAGE_DIMENSION` 并重新编码为 JPEG（`_prepare_image`），再 base64 发送
- 一次 Vision 调用（`_build_prompt` / `_call_vision`）只要求模型做两件事：（1）给 6 个身体部位区域给出边界框——`REGION_KEYS = (head, torso, right_arm, left_arm, right_leg, left_leg)`，每个区域要么 `visible: true` + `bbox: [x0, y0, x1, y1]`（0~1 的比例坐标），要么 `visible: false`；（2）从固定枚举里选 `hair_style`（`HAIR_STYLES`）和 `eye_shape`（`EYE_SHAPES`）两个分类字段。模型完全不再输出任何颜色或像素网格
- `_validate_regions_response` 校验结构：`hair_style`/`eye_shape`/`regions` 整体缺失或格式错误 → 整个响应判定不完整、重试（最多 `max_retries=2` 次，全部失败则退化为全部区域不可见 + 默认发型/眼型）；单个区域自己的 bbox 格式不对，只会让那一个区域退化成"不可见"，不会拖累整个响应重试
- `ai_model`（`"flash"` / `"flash-lite"`）通过 `_resolve_model_name` 映射到 `GEMINI_MODEL_FLASH` / `GEMINI_MODEL_FLASH_LITE`，逐请求可选，方便对比两个模型的效果，返回的 `metadata.ai_model` 会回显实际用的是哪个
- `_render_photo` 是真正产出像素的地方：它把定位到的区域交给 `photo_render.py` 做确定性图像处理，产出 6 个"正面"网格（`head_front` + `body_front` + 双臂双腿正面）的 hex 像素 + 原始索引网格 + 共享调色板 + 派生出的具名颜色（`skin_tone`/`hair_color`/`shirt_main`/`arm_main`/`pants_main` 等，供 `procedural.py` 用）
- `decode_indexed_grid` 仍然是"索引网格 → hex 网格"的工具函数，只是现在喂给它的索引来自 `photo_render.quantize_shared` 的量化结果，而不是 AI 直接输出的 JSON

### Photo-Derived Rendering (`app/services/photo_render.py`)
确定性的图像处理模块（新增），不调用任何 AI，只用 Pillow 处理真实上传的照片，按 `claude_vision._render_photo` 传入的区域框产出像素：
- `crop_region` 把照片按边界框裁剪出来（越界会被 clamp，裁剪结果零宽高时返回 `None`）
- `downsample_dominant` 把裁剪区域降采样到目标皮肤面的行列数，每个格子取"该格源图块里最常见的颜色"（而不是平均色），这样图案边界（比如红色滚边贴白色底）不会被平均糊成过渡色
- `quantize_shared` 把 torso + 四肢正面这 5 个部位的所有像素合并成**一次**量化（`max_colors=24`），保证同一皮肤同一颜色（比如同一块肤色）在手臂和腿上量化出同一个色号，而不是各自漂移
- 脸部单独处理，不走 `quantize_shared`：`extract_face_colors` 用 `head_front` 固定的 8 行模板（第 0-1 行头发、2-3 行眼睛、第 4 行皮肤、5-6 行嘴、第 7 行脖子）在照片对应区域的每个"行带"里采样真实像素——皮肤/头发取该行带的主色（`_dominant_color`），眼睛/嘴唇取该行带量化成 2 色后的"少数色"（`_minority_color`，因为五官在行带里总是占少数面积）；`build_head_front` 再把这些测得的颜色套进固定模板拼出 8×8 的 `head_front` 网格（`eye_shape` 决定眼睛画 1px 还是 2px 宽）
- 如果某个区域模型判定不可见（或裁剪退化成零尺寸），`claude_vision._render_photo` 会用同名兄弟部位已测出的颜色，或 `procedural.DEFAULT_COLORS` 里的默认色兜底填充，保证 6 个正面网格永远齐全，不会在拼装时留空洞

### Procedural Fill (`app/services/procedural.py`)
只有 `AI_GENERATED_KEYS` 这 6 个正面面（`head_front`、`body_front`、`right_arm_front`、`left_arm_front`、`right_leg_front`、`left_leg_front`）是照片/AI 定位出来的，其余基础区域（body 的 back/top/bottom/left/right、双臂各自的 back/top/bottom/left/right、双腿各自的 back/top/bottom/left/right、head 除 front 外的 5 个面）**不经过 AI**，由 `generate_procedural_regions` 程序化生成，不做任何合成阴影（Minecraft 自己的光照已经会给 3D 模型加阴影）：
  - back/left/right 用 `propagate_front_row` 直接复制同一部位正面网格里每一行中间列的颜色，所以正面上一条衣服边界（比如袖子在半路结束）沿着肢体包裹一圈后依然连贯
  - top/bottom 这两个小端面、以及 head 除 front 外的 5 个面，用具名颜色（`_render_photo` 派生出的，或 `DEFAULT_COLORS` 兜底）做纯色填充（腿的 bottom 用 `shoe_color`，其余用该部位的 `_main` 色 / `head_fill_color`）
  左右两侧直接复用同一套规则（不需要真正的像素镜像，因为规则本身就是对称的）；`exclude_keys` 参数让已有真实数据的正面面不被程序化覆盖。这是 token 优化的核心：AI 一次调用只返回 6 个区域框 + 2 个分类字段（远小于当年逐像素输出的量），真正的像素全部由 `photo_render.py` 确定性生成。

### Skin State Persistence (`app/services/skin_store.py`)
生成时仍然把完整调色板 + 每个正面面的原始索引网格（不是解码后的 hex）存成 `skins/<id>.json`，和 PNG 放在一起（`save_skin_state`，在 `routers/skin.py` 的 `POST /api/generate` 里调用）。这套持久化是为**可能的未来**对话式改色功能保留的——之前那版对话式改色（`POST /api/skin/{id}/edit`）依赖固定角色调色板，随该设计一起被移除了；`load_skin_state` 目前在生产代码里没有任何调用方，纯粹是为将来重新设计编辑功能预留的数据。

### Skin UV Map (`app/services/skin_map.py`)
所有 Minecraft 64×64 皮肤 UV 坐标，以 `FaceRect(x, y, w, h)` dataclass 表示。`PIXEL_KEY_MAP` 将模型输出的 key（如 `"head_front"`）映射到区域组 + 面名。`get_all_regions(model)` 根据 Classic/Slim 返回正确坐标集。

### Skin Assembler (`app/services/skin_assembler.py`)
接收像素数据 dict（AI 解码结果 + procedural 结果合并后的完整 hex 颜色网格）→ 用 Pillow 创建 64×64 RGBA 图片，在正确的 UV 坐标处绘制每个面。验证网格尺寸，静默跳过格式错误的数据。这一层的接口没有变化，仍然只认 `{key: 2D hex 数组}`。

### API Layer (`app/routers/skin.py`)
- `POST /api/generate` — 接收 multipart 图片上传 + model 类型 + 可选 `ai_model`（`"flash"`/`"flash-lite"`，缺省用 `GEMINI_DEFAULT_MODEL`），返回 skin_id + skin_url，同时把生成状态存进 `skin_store`
- `GET /api/skin/{id}.png` — 从 `skins/` 目录提供生成的 PNG

### Frontend (`app/static/`)
纯 HTML/CSS/JS，无构建步骤。使用 skinview3d@3.1.0（CDN）做 3D 皮肤预览（行走动画）。拖拽上传 + Classic/Slim 切换。

## Key Design Decisions

- **OpenAI 兼容 SDK**: 使用 `openai` Python 包，通过 `base_url` 指向 Gemini 的 OpenAI 兼容端点，方便将来切换其他兼容服务
- **逐请求切换模型**: `ai_model=flash` / `flash-lite` 让两个免费额度模型可以不重启服务直接对比效果
- **Python 3.9 compatibility**: 所有文件使用 `from __future__ import annotations`
- **No database**: 生成的皮肤以 UUID 文件名保存在 `skins/` 目录
- **绝大部分程序化/确定性生成**: AI 只定位 6 个区域框 + 分类两个字段，不画任何像素；6 个正面面的像素由 `photo_render.py` 从真实照片裁剪/降采样/量化得到，其余全部由 `procedural.py` 用 `propagate_front_row`（back/left/right 复制正面每行颜色）+ 端面纯色填充规则生成，这是 token 消耗和保真度的主要优化点
- **Retry logic**: 单次调用响应不完整（`hair_style`/`eye_shape`/`regions` 结构缺失或格式错误）时整体重试，最多 2 次；单个区域框自己格式错误只会让那个区域退化为不可见，不触发整体重试
- **Classic vs Slim**: 唯一区别是手臂 front/back 宽度 4→3px，由 `skin_map.py` 坐标选择处理
