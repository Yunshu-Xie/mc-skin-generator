# mc-skin-generator

照片 → Minecraft 皮肤。上传一张图，得到一张 **128×128** 的皮肤 PNG（外加一份 64×64 兜底，因为原版 Java 只收这个尺寸），附带浏览器内 3D 预览和一键下载。

关键在于分工：**AI 只判断语义**（脸、眼睛、上身、四肢在照片里的位置，以及各材质的颜色），**像素全部由一条真实的图像处理管线从照片本身推导**——线性光、OKLab、结构保持降采样、全局联合调色板量化。

技术栈：FastAPI · numpy · Pillow · 原生 HTML/JS（无构建步骤）· 通过 OpenAI 兼容客户端调用 Gemini。

设计原因写在 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**。

![各降采样方法对比](docs/method-comparison.png)

## 为什么不让 AI 直接画像素

早期版本让模型逐像素输出 head 六个面和 body_front 的调色板索引（480 个格子）。做不好，而且原因不在 prompt：**语言模型没有像素级空间精度**，它无法可靠地把一只眼睛放进第 2 行第 2 列。

现在模型回答的是"东西在哪、是什么颜色"——大约 20 个数字。更便宜、更快、几乎不再重试，而像素质量由可测试的数值代码决定。

## 特性

- **照片 → 皮肤**：JPEG / PNG / WebP / GIF（≤ 5 MB）
- **色彩正确**：一切在线性光和 OKLab 里计算。不在 sRGB 数值上求平均（那是最经典的降采样 bug），不用 HSV 当明度
- **按内容选降采样**：脸用 DPID（保住五官），衣服四肢用众数色（保住干净的色块边界，白衬衫上的红条纹不会糊成粉）
- **一张调色板管全身**：材质在各部位之间保持一致；前 7 个槽是固定的语义角色
- **免费改色**：`POST /api/skin/{id}/recolor` 换掉一个调色板颜色——不调 AI、不重新渲染
- **自带质量指标**：每次生成返回 SSIM / ΔE / 细节保留，调参不靠感觉
- **两种分辨率**：128×128 为主（Bedrock 原生／Java 装 HD Skins 或 CustomSkinLoader），每次同时导出 64×64 兜底。分辨率还决定算法——8×8 的脸必须绘制，16×16 的脸改走照片重采样
- **Classic & Slim**、**3D 预览**（[skinview3d](https://github.com/bs-community/skinview3d)）、**直接下载**
- **离线可跑**：没有 API Key 时用默认版式，仍能跑通整条管线

## 快速开始

```bash
git clone https://github.com/Yunshu-Xie/mc-skin-generator.git
cd mc-skin-generator

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env      # 填入 GEMINI_API_KEY（留空也能跑，走默认版式）
uvicorn app.main:app --reload
```

打开 http://127.0.0.1:8000

## 调参

```bash
# 不花 API 额度，用内置的合成人像看三种降采样方法的差别
python3 tools/compare.py --demo --eyes 0.385,0.210,0.615,0.250

# 自己的照片，手工给出两个关键框
python3 tools/compare.py photo.jpg --face 0.38,0.05,0.66,0.27 --eyes 0.43,0.16,0.65,0.19

# 走真实 vision 调用
python3 tools/compare.py photo.jpg --ai
```

**同时看数字和图。** 这个尺度下 SSIM 会误导你：它被大片平坦区域主导，一个把眼睛平均掉、把脸颊做准的方法反而得分更高。理由和真正的解法见 ARCHITECTURE §7 与 §6。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/generate` | multipart 上传 + `model`（classic/slim）+ 可选 `ai_model`。返回 `skin_id` / `palette` / `roles` / `metrics` |
| `POST` | `/api/skin/{id}/recolor` | `{"old_color": "#3B5998", "new_color": "#B03030"}`。不调 AI |
| `GET` | `/api/skin/{id}.png` | 下载主贴图；`?vanilla=1` 取 64×64 兜底 |

## 测试

```bash
pytest           # 132 项，含色彩科学的数值回归测试
ruff check .
```

其中 `tests/test_color.py::test_averaging_in_srgb_is_wrong` 把 gamma 这件事钉成了回归测试：黑与白的正确中间调是 sRGB 188，不是 128。
