# DiffSynth 模型支持度验证（昇腾实测）—— 任务总结

> **一句话**：在 8 卡昇腾 950PR 上，把 DiffSynth 支持的各模型逐个真跑一遍：10 个模型里
> **7 个完全跑通**、1 个受限（FLUX 只能用 bf16）、2 个跑不了（sd3 框架无管线、
> Qwen-Image 2.1 结构未适配）。
>
> 验证周期：2026-10-31 ~ 11-02　　环境：8 × Ascend950PR / CANN 9.2.0
> 判定口径：**只看端到端能否跑通（加载 → 推理 → 出文件），精度不纳入判定**

---

## 一、任务是什么

框架层面统计只能回答"代码里写了支持哪些模型"，回答不了"到底能不能跑"。这次是**上机实测**：

> 给一批模型名，逐个在昇腾 NPU 上跑真实推理，看能不能产出图片 / 视频 / 3D 网格。

**判定标准只有一条**：产物是否成功生成、进程是否异常退出。

- 不评估精度、性能、画质，不做参数调优
- 文中出现的 SSIM 数值只作记录，**不作为判定依据**
- 所以标为"SSIM 未达标"的用例，**仍然算跑通**

这个口径很关键——否则看到 0.70 的相似度会误判成失败，实际上图是正常出来的，只是和基准图数值有差异。

---

## 二、结论：10 个模型，7 个完全跑通

### 2.1 总数

用户给出 12 个名称，实际对应 **10 个独立模型**（`text_encoder` / `vae` 是依附主模型的子模块，不单独构成模型）。

| 状态 | 数量 | 模型 |
|---|---|---|
| ✅ **完全跑通** | **7** | sd、sdxl、z_image、flux2、qwen_image（初代）、wan、hunyuan3d |
| ⚠️ 受限跑通 | 1 | flux（bf16 可跑；fp8 硬件不支持、fp16 溢出全白） |
| ❌ 跑不了 | 2 | sd3（框架无推理管线）、Qwen-Image 2.1（结构不匹配） |

**换算下来：8 / 10 能在昇腾上出结果**，其中 7 个无任何限制。

### 2.2 逐模型明细

| # | 模型 | 权重体积 | 验证方式 | 耗时 | 结果 |
|---|---|---|---|---|---|
| 1 | **sd** (1.5) | 4.0 G | 自写最小脚本，512²/20 步 | ~20 s | ✅ 出图，54 it/s |
| 2 | **sdxl** | 6.5 G | 官方 pytest，1024²/20 步 | 21 s | ✅ 2 passed / 2 failed（失败属精度与上游缺陷） |
| 3 | **z_image** | 31 G | 官方 pytest，1024²/9 步 | 15 s | ✅ 图正常，SSIM 0.7646 |
| 4 | **flux2** (klein 4B) | 23 G | 官方 pytest，1024²/4 步 | 6.9 s | ✅ 出图，SSIM 0.7081 |
| 5 | **qwen_image** (20B) | 54 G | 官方 pytest，1328²/28 步 | 53 s | ✅ 出图，SSIM 0.8656 |
| 6 | **wan** (2.1 视频) | 17 G | 官方 pytest，41 帧 480² | 4 m 55 s | ✅ **1 passed**，产出 mp4 |
| 7 | **hunyuan3d** | 6.9 G | 官方 pytest，50 步 | 1 m 02 s | ✅ **1 passed**，产出 glb |
| 8 | flux (1.0) | 16 G | 三种精度分别验 | — | ⚠️ bf16 可跑 / fp8 崩 / fp16 全白 |
| 9 | sd3 | — | 无 `sd3_image.py` | — | ❌ 框架未提供推理入口 |
| 10 | qwen_image3 (2.1) | 30.9 G | 复用 QwenImagePipeline | — | ❌ 结构不匹配，加载失败 |

产物齐全：`sd.png`、`sdxl.png`、`z_image.png`、`flux2.png`、`qwen_image.png`、
`wan_t2v.mp4`、`hunyuan3d.glb`、`flux_bf16.png`。

---

## 三、怎么测的（可复现）

### 3.1 用官方测试，一行不改

核心做法：**跑框架自带的 pytest 用例**，仅加一个兼容层把 `cuda` 映射成 `npu`。

```python
# tests/conftest.py —— 唯一新增的文件
# 把测试里的 device="cuda" 映射到 npu，测试逻辑零改动
```

这样测出来的结论是"框架自己认为该跑通的，在昇腾上能不能跑通"，而不是"我另写一套代码能不能跑"。

例外两处：

- **SD 1.5**：框架没有对应的 pytest 用例，另写了最小脚本（调用同一套 pipeline、同一份权重）
- **FLUX.1**：官方用例绑定 fp8 精度，而昇腾不支持 fp8，改用自写脚本对比 bf16 / fp16

### 3.2 四个必须的设定（缺一即失败）

```bash
export DIFFSYNTH_CACHE=/models/ds/cache              # 权重目录（157 GB）
export PYTHONPATH=/models/ds/code/DiffSynth-Engine   # 代码目录（未 pip install）
export MS_HUB_OFFLINE=1                              # 只读本地
export ASCEND_RT_VISIBLE_DEVICES=6                   # 指定卡
/models/ds/venv/bin/python ...                       # 必须用 venv 的 python
```

两个必踩的坑：

| 现象 | 根因 |
|---|---|
| `ServerError [E1002] unihub GET ... http=401` | 没设 `MS_HUB_OFFLINE=1`。框架每次启动都去 ModelScope 校验目录树，接口未鉴权返回 401——**与本地网络无关**，很容易误判成网络问题 |
| `create_causal_mask() got an unexpected keyword argument 'input_embeds'` | 用了系统的 `python3`（容器的 transformers 5.13.0 把形参改名了）。venv 固定 `4.57.1` |

---

## 四、失败不等于"跑不了"——四类根因要分清

这是本次验证最有价值的部分：**把"失败"拆成四种，只有后两种才是真问题**。

### A. 精度不达标（不影响"能跑"）

| 用例 | 实测 | 阈值 |
|---|---|---|
| sdxl `test_fused_lora` | 0.98881 | 0.99 |
| z_image `test_txt2img` | 0.7646 | 0.99 |
| flux2 `test_txt2img` | 0.7081 | 0.95 |
| qwen_image `test_txt2img` | 0.8656 | 0.99 |

图都正常生成了，差异来自 NPU 数值路径 + 采样步数偏少（flux2 只跑 4 步）。

### B. 上游测试用例自身缺陷（与 NPU 无关）

`sdxl test_inpainting`：`F.interpolate` 收到 5 维张量报错。

根因是测试基类把灰度 mask 强制 `.convert("RGB")` 变成 3 通道，`prepare_mask()` 又按单通道做两次
`unsqueeze`，得到 `(1,1,H,W,3)`。**这个缺陷在 CUDA 上同样会失败**，不是昇腾的问题。

### C. 平台硬限制：fp8 计算不可用

`FLUX.1-dev-fp8`（T5 与 DiT 都是 fp8）：

| 精度 | 结果 | 报错 |
|---|---|---|
| fp8 原生 | ❌ | `aclnnAddmm ... error code 161002`，`Tensor matB not implemented for DT_FLOAT8_E4M3FN` |
| bf16（升精度） | ✅ | 出图正常，但画面发虚——权重本身只有 3 位尾数，信息已丢失 |
| fp16（升精度） | ⚠️ | 跑完但全白，激活值溢出成 inf |

**结论**：昇腾上跑 FLUX 只有 bf16 一条路，且应该用原生 bf16 权重，而不是 fp8 升精度。

### D. 框架能力缺口（真正的"跑不了"）

**D1. sd3：有模型、无入口**

`models/sd3/` 有模型结构定义，但 `pipelines/` 下没有 `sd3_image.py`：

```
ImportError: cannot import name 'SD3ImagePipeline' from 'diffsynth_engine.pipelines'
```

即使下载 66.7 G 的 SD3.5-large 也跑不起来。

**D2. Qwen-Image 2.1：结构已变，未适配**

| 组件 | 初代 | 2.1 |
|---|---|---|
| 文本编码器 | hidden 3584 / 28 层 / vocab 152064 | hidden 4096 / 36 层 / vocab 151936 |
| DiT 主干 | 60 层 / 24 heads / patch_size 2 | 32 层 / 32 heads / patch_size 1 |

框架内结构参数**写死为初代**，不读仓库自带的 `config.json`，所以加载即失败。

---

## 五、和"代码级统计"的差异（重要）

同期还有一份代码级统计（`2026 昇腾框架模型支持度（NV与NPU）.xlsx`），结论是：

> DiffSynth 在 NPU 上只有 **1 个**模型有昇腾痕迹（FLUX.2，因为 `flux2_dit.py` 里有
> `device.type == "npu"` 分支）。

而这次实测是 **7 个跑通**。两个数字差这么多，原因很简单：

| | 代码级统计 | 实测 |
|---|---|---|
| 回答的问题 | 仓库里有没有**专门为昇腾写的代码** | 模型能不能**真的跑出结果** |
| 结果 | 1 个 | 7 个 |

**DiffSynth 的模型实现大部分是设备无关的纯 PyTorch**，只要有一层 `cuda → npu` 的映射就能跑，不需要为昇腾单独写分支。所以代码级统计会**严重低估**实际可跑性。

这个差异不是谁错了，而是两个口径回答不同问题。但对外汇报时说清楚很重要——否则"统计说 1 个、实测说 7 个"看起来像矛盾。

---

## 六、产出物

| 类别 | 内容 |
|---|---|
| 验证报告 | `模型支持度验证报告.md`（10 节，含逐模型明细、四类根因、复现命令） |
| 出图产物 | 7 个模型的产物文件（png / mp4 / glb），服务器 `/data/disk1/models/ds/out/`，本地有副本 |
| 验证脚本 | `smoke_sd.py`（SD 最小脚本）、`flux_prec.py`（FLUX 三精度对比）、`run_one.py`（逐模型取证） |
| 兼容层 | `tests/conftest.py`（cuda → npu 映射，官方测试零改动） |
| 权重 | 157 GB，缓存在 `/data/disk1/models/ds/cache` |

---

## 七、结论与建议

### 三条结论

1. **框架的模型支持是"一个家族一条链路"**：每个模型族需要 Config 类 + StateDictConverter（权重名映射）+ Pipeline 三件套，**只有权重跑不起来**。所以盘点支持度必须逐家族验证，"跑通一个"不能代表全部。
2. **本版框架在昇腾上支持 8 / 10**，缺口集中在两类：上游没补的管线（sd3）、结构已变但未适配的新版本（Qwen-Image 2.1）。
3. **真正的平台级限制只有一条：fp8 计算不可用**。其余失败都是精度或上游用例缺陷，不构成"跑不起来"。

### 三条建议

| 建议 | 理由 |
|---|---|
| 把 `conftest.py` 和 venv 的 `transformers==4.57.1` 固化进镜像 | 每个新人都会重踩这两个坑 |
| Qwen-Image 2.1 若要支持，把写死的结构参数改为读 `config.json` | 否则每出新版本都要改代码 |
| FLUX 若作为正式支持项，改用原生 bf16 权重 | fp8 升精度只是"让算子算得动"，画质已经损失 |

---

## 八、一句话总结

> 12 个名称收敛成 10 个模型，逐个上机真跑：**7 个完全跑通、1 个只能 bf16、2 个跑不了**。
> 真正的平台限制只有 fp8 一条，其余全是精度差异或上游缺管线——**这个区分比"跑通几个"本身更有价值**。
