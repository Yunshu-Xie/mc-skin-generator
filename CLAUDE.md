# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

照片 → Minecraft 皮肤：上传图片，AI **只**判断语义（各部位在照片里的 bbox + 角色颜色），像素全部由一条真实的图像处理管线从照片本身推导，输出 64×64 PNG + 3D 预览 + 下载。FastAPI 后端，原生 HTML/JS 前端。

**读代码前先读 `docs/ARCHITECTURE.md`** —— 它解释了每个设计决策背后的原因，这里只列结构。

一句话版：旧管线让模型逐像素画 8×8 的脸，而语言模型没有像素级空间精度，这是"精细度上不去"的根本原因；现在模型只回答"东西在哪、是什么颜色"，像素由 `app/imaging` 计算。

## Common Commands

```bash
# 安装依赖
pip install -e ".[dev]"          # 或：pip install fastapi uvicorn openai Pillow numpy python-multipart pydantic-settings pytest httpx ruff

# 开发服务器
uvicorn app.main:app --reload

# 测试
pytest
pytest tests/test_renderer.py -v
pytest tests/test_downscale.py::test_dpid_keeps_an_outlier_that_box_averages_away -v

# 渲染质量对比（不花 API 额度）
python3 tools/compare.py --demo --eyes 0.385,0.210,0.615,0.250
python3 tools/compare.py photo.jpg --face 0.38,0.05,0.66,0.27 --eyes 0.43,0.16,0.65,0.19
python3 tools/compare.py photo.jpg --ai     # 走真实 vision 调用

# Lint
ruff check . && ruff format .
```

## Configuration

`.env`：

- `GEMINI_API_KEY` — Google AI Studio 的 API Key（https://aistudio.google.com/apikey）。**留空则整条管线走离线默认版式**，仍然可以跑通，方便本地开发和测试
- `GEMINI_BASE_URL` — Gemini 的 OpenAI 兼容端点
- `GEMINI_MODEL_FLASH` / `GEMINI_MODEL_FLASH_LITE` / `GEMINI_DEFAULT_MODEL` — 逐请求用 `ai_model` 字段（`"flash"` / `"flash-lite"`）选择
- `MAX_IMAGE_DIMENSION` — 发给 vision 前压缩到的最长边（默认 768）
- 渲染参数：`PALETTE_SIZE`（16）、`FACE_METHOD`（`dpid`）、`MATERIAL_METHOD`（`dominant`）、`PRESHARPEN`（0.45）、`DPID_LAMBDA`（1.4）

`app/config.py` 通过 pydantic-settings 读取。

## Architecture

### `app/imaging/` —— 纯图像科学
无 AI、无 Web、无 Minecraft，进出都是 numpy 数组，全部可数值测试。

- **`color.py`** — sRGB ↔ **线性光** ↔ **OKLab**，ΔE，hex 边界转换。这一层之后没有任何代码在 sRGB 数值上做平均——在编码值上求平均没有数学意义，这是最经典的降采样 bug
- **`downscale.py`** — `box`（精确面积平均）、`dpid`（离本格均值越远权重越大，OKLab 距离）、`dominant`（众数色）；`unsharp` 预锐化补偿 16× 缩小的 MTF 损失；`fit_crop` 裁剪而非拉伸
- **`quantize.py`** — OKLab 里的加权 k-means（k-means++ 初始化，支持固定中心）；`Palette` 的前若干槽是**锚定的语义角色**，永不被优化器移动
- **`metrics.py`** — SSIM（算在 OKLab 的 L 通道上）、ΔE、`detail`。**这些指标都是面积加权的，没有一个能判断"眼睛还在不在"**，见 ARCHITECTURE §7

### `app/services/layout.py` —— AI 层（语义，不产像素）
一次 vision 调用返回：`face` / `eyes` / `hair_top` / `torso` / `right_arm` / `left_arm` / `legs` 的归一化 bbox + 7 个角色颜色 + 一句描述。输出约 20 个数字。`analyze_photo` **永不抛异常**：网络失败、JSON 坏掉、字段缺失都退化成默认版式。

### `app/services/renderer.py` —— 管线编排
解码到线性光 → 按 bbox 裁剪（`expand_to_aspect` 把框**扩大**到目标宽高比，不是裁掉，否则头顶和下巴会被切掉）→ 预锐化 → 按面选降采样方法（脸用 `dpid`，衣服四肢用 `dominant`）→ **一次全局加权量化**建立唯一调色板 → 补齐照片拍不到的面 → 全部吸附到该调色板 → 打分。

`stamp_eyes` 是唯一一处覆盖降采样结果的语义先验，用眼部 bbox 作显著性掩码而不是直接涂色带；`recolor` 因为"整张皮肤共用一张调色板"而只是字符串替换。

### `app/services/shading.py` —— 程序化面
背面/内侧从**同一张调色板**取色，按面朝向做 OKLab 明度偏移 + 柔和的上下渐变。旧的 `flat_fill_with_border`（1px 深色描边）被删掉了——那让每条肢体在游戏里读起来像一个画出来的方框。

### `app/services/skin_map.py` / `skin_assembler.py`
64×64 UV 坐标表与 PNG 组装。**接口未改动**，仍然只认 `{key: 2D hex 数组}`。

### API (`app/routers/skin.py`)
- `POST /api/generate` — multipart 上传 + `model`（classic/slim）+ 可选 `ai_model`；返回 skin_id、palette、roles、metrics
- `POST /api/skin/{id}/recolor` — 换一个调色板颜色，不调 AI、不重渲染
- `GET /api/skin/{id}.png`

## Key Design Decisions

- **AI 不画像素**：语言模型没有像素级空间精度。这是整个重构的前提
- **一切在线性光 / OKLab 里做**：不在 sRGB 上平均，不用 HSV 表示明度
- **按内容选降采样方法**：脸要特征存活（`dpid`），衣服要色块干净（`dominant`）
- **一张调色板管全身**：材质一致性 + 让改色变成 O(1) 操作
- **语义角色锚定固定槽位**：0=skin_tone … 6=shoe_color
- **离线可跑**：没有 API Key 时用默认版式，测试与本地开发不需要网络
- **Python 3.11+**，所有文件 `from __future__ import annotations`
- **无数据库**：皮肤以 UUID 文件名存在 `skins/`（PNG + 同名 JSON）
- **Classic vs Slim**：唯一区别是手臂 front/back 宽度 4→3px，由 `skin_map.py` 处理

## 改动这个项目时

- 动了 `app/imaging` 任何数值行为 → 先加数值测试，再改实现
- 调渲染参数 → 用 `tools/compare.py`，**同时看数字和图**；只看 SSIM 会把你带向"糊掉眼睛"的方向
- 想加新的语义先验（嘴、眼镜、logo）→ 先读 ARCHITECTURE §5 和 §6，考虑它是不是应该由联合优化自动得出
