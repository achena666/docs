# 昇腾 NPU 从零部署推理模型 · 完整决策与操作指南

> 适用场景：拿到一台昇腾服务器，只知道机器型号（如 Atlas 800T A2 / 910B 服务器），
> 需要从零部署一个推理模型（LLM、扩散模型、多模态等）。
> 本指南来自 Wan2.1-T2V 在 910B3 上从零部署的实战复盘。

---

## 0. 先建立心智模型：昇腾的软件栈

```
┌─────────────────────────────┐
│  你的模型 + 推理代码          │  ← 模型层
├─────────────────────────────┤
│  MindIE / vllm-ascend /     │  ← 推理框架层（可选，看模型类型）
│  torch_npu + PyTorch        │
├─────────────────────────────┤
│  CANN (toolkit + kernels)   │  ← 算子层，昇腾的"CUDA"
├─────────────────────────────┤
│  驱动 + 固件 (npu-smi)       │  ← 最底层，一般机器出厂已装
└─────────────────────────────┘
```

**核心铁律：四层版本强绑定**。驱动决定 CANN 版本，CANN 决定 torch_npu/MindIE 版本，
torch_npu 又绑定 torch 版本。任何一层乱动都会 symbol error / 算子不支持。
部署 90% 的坑来自版本配套，所以第一步永远是查配套表。

---

## 1. 阶段一：环境摸底（只知机器型号时的第一步）

在服务器上逐条执行，把结果记下来：

```bash
# ① NPU 型号、卡数、显存、占用情况（最关键）
npu-smi info

# ② 驱动和固件版本（查配套表的输入）
npu-smi -v
cat /usr/local/Ascend/driver/version.info

# ③ 宿主机已装的 CANN / MindIE
ls /usr/local/Ascend/
cat /usr/local/Ascend/ascend-toolkit/latest/*/ascend_toolkit_install.info

# ④ 系统、Python、Docker
cat /etc/os-release | head -3; uname -m
python3 --version
docker --version; docker images | head -20   # 重点看有没有 ascend/cann/mindie 系镜像

# ⑤ 磁盘空间（权重很占地方，LLM 动辄几十 GB）
df -h / /home

# ⑥ 谁在用卡、有没有别人占资源（共用机器必查）
ps -ef | grep -Ei "vllm|mindie|python" | grep -v grep
```

摸底要回答四个问题：

1. **什么卡？** 310P（推理卡，显存小）→ 只能跑小模型；910A/B（训练+推理）→ 都行；
2. **驱动多新？** → 决定能用哪个版本的 CANN（查配套表）；
3. **有没有现成环境？** 宿主机 CANN/容器镜像里常有前人装好的东西，能复用就别重装；
4. **机器是不是共用的？** 共用 → 必须容器隔离 + 避开被占用的卡 + 不要 kill 别人进程。

---

## 2. 阶段二：决策——选哪条部署路线

先问一个问题：**你要部署的模型，官方适配了吗？** 按优先级从高到低选路线：

### 路线优先级决策树

```
要部署的模型是什么？
│
├─ 大语言模型（Qwen/Llama/DeepSeek...）？
│   └─ 首选 vllm-ascend 容器镜像（昇腾官方维护，兼容性最好）
│       或 MindIE-LLM（要 OpenAI 兼容服务、国密合规时）
│
├─ 扩散/多模态/其他（Wan、SD、Qwen-VL...）？
│   ├─ 官方/社区有预适配镜像或代码包？（搜 "模型名 + ascend/MindIE"）
│   │   └─ 有 → 直接用，零补丁（最优解）
│   ├─ MindIE 套件支持该模型且有 root？
│   │   └─ 有 → 装 MindIE + 官方适配代码（次优，要 root）
│   └─ 都没有 → 通用 torch_npu 容器 + 手工适配（下策，参考本指南第 4 节）
│
└─ 冷门/自研模型？
    └─ 只有下策：torch_npu + 手工适配，做好踩坑准备
```
注意：
1.MindIE-LLM，vllm-ascend这两个都是深度绑定 CANN、做了 NPU 底层算子融合优化的 LLM 推理引擎，
MindIE-LLM是华为推理引擎专门处理大语言模型的模块是针对昇腾的这套硬件从零开发的，闭源
vllm-ascend是在vllm的基础上，根据华为昇腾这个硬件的特性，做了部分的调整

2.预适配镜像：把“操作系统 + CANN + torch_npu + 适配好的模型代码 + 依赖库”全部装好，打包成一个 Docker 镜像。你拉下来 docker run 就能跑，什么都不用装、什么都不用改。

3.预适配代码包：MindIE 版的 Wan2.1 仓库，华为的工程师已经把 RoPE 复数问题、NPU 注意力这些全改好了，准备好环境clone 下来配上权重就能跑

4.MindIE 这类官方套件必须安装到系统目录（/usr/local/Ascend/），这个目录普通用户写不进去，必须 root 才能动。


### 三条路线对比

| 路线 | 适用 | 前置条件 | 工作量 | 性能 |
|---|---|---|---|---|
| A. 官方预适配镜像/代码 | 模型被官方适配过 | docker 权限 | 改个路径就跑 | 优化过，最好 |
| B. MindIE 套件 | MindIE 支持该模型 | root + 宿主机安装 | 中等，走官方安装文档 | 优化过 |
| C. 通用 torch_npu 容器 + 手工适配 | 什么都没有 | docker 权限 | 大，要打补丁调试 | 通用实现，略慢 |

**原则：能用现成的就不要自己造。** 每少一层自己动手的东西，就少一批坑。

### 怎么查"官方适配了没"

- 魔乐社区 modelers.cn / 昇腾社区 hiascend.com 搜模型名；
- Gitee 的 ascend 组织下搜（gitee.com/ascend/）；
- ModelScope/HuggingFace 上搜模型名 + "ascend"、"npu" 标签；
- MindIE 支持列表查昇腾官方文档《MindIE 支持的模型清单》。

---

## 3. 阶段三：标准部署流程（以最常见情况为例）

### 3.1 查版本配套表（永远的第一步）

到昇腾官方文档《CANN 版本配套表》，按 驱动版本 → CANN 版本 → torch_npu 版本 → torch 版本 逐层确定。
把四个版本号写下来再动手，别边装边定。

### 3.2 起容器（推荐所有路线都用容器）

```bash
mkdir -p ~/deploy/{code,weights,output}

docker run -dit --name mymodel \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v ~/deploy:/workspace \
  <镜像名>:<tag> bash

docker exec -it mymodel bash
```

要点：
- `--device` 只透传你用的那张卡（`/dev/davinci0` = NPU 0），多卡就多写几行；
- 宿主机的 driver/dcmi/npu-smi 必须挂进去（容器只带 CANN 不带驱动）；
- 数据目录挂载到家目录，容器删了数据不丢；
- 多人共用机器时，容器是唯一安全的隔离方式。

**进门自检三连（每次都要跑）：**
```bash
python3 -c "import torch, torch_npu; print(torch.__version__, torch.npu.is_available(), torch.npu.device_count())"
```
四个版本信息都在这一条命令里验证了，不通过就先修环境再往下走。

### 3.3 下权重 + 装依赖

```bash
# 权重优先国内源（ModelScope），大文件支持断点续传
modelscope download --model <org>/<model> --local_dir /workspace/weights/<model>

# 下载完必须校验体积（对官方文件清单，防止半截文件）
du -sh /workspace/weights/<model>

# 依赖：先读 requirements.txt 甄别，再装
cat requirements.txt   # 找出 CUDA 专用包（flash_attn/xformers/triton）并剔除
pip install <推理必需的包> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**装完 pip 必查 torch 是否被偷换**（依赖解析经常拉降级）：
```bash
python3 -c "import torch; print(torch.__version__)"
```

### 3.4 跑推理（长任务规范）

```bash
# 长任务一律后台化，SSH 断线零损失
nohup python generate.py --参数 --save_file output/result.mp4 > run.log 2>&1 &

# 看进度（Ctrl+C 只退出观看，不杀任务）
tail -f run.log
```

要点：
- 输出文件**必须带完整扩展名**（很多工具靠扩展名识别格式）；
- 显存不够时优先考虑 `--offload_model` / `--t5_cpu` 之类的卸载参数，而不是换卡。

### 3.5 验收与固化

```bash
ls -lh /workspace/output/                     # 产物在不在
docker commit mymodel mymodel:v1              # 环境固化成镜像快照
```

固化后，下次重装只要：起容器（用 v1 镜像）→ 跑推理，5 分钟出活。

---

## 4. 下策路线（通用 torch_npu + 手工适配）的典型补丁清单

当模型没被官方适配时，NVIDIA 代码跑昇腾通常要过这几道门（按出现顺序）：

| 症状 | 根因 | 补丁方向 |
|---|---|---|
| `Torch not compiled with CUDA enabled` | 代码调 `torch.cuda.*` | 入口注入 `import torch_npu; from torch_npu.contrib import transfer_to_npu` |
| `dtype Complex128 not in support list` | NPU 算子不支持复数 | 把复数运算改写为等价实数运算（如 RoPE 的 cos/sin 分解） |
| `assert q.device.type == 'cuda'` | 设备断言写死 | 断言放行 `'npu'` |
| `assert FLASH_ATTN_2_AVAILABLE` | flash_attn 是 CUDA 专用包 | 注意力函数转调 torch 原生 SDPA |
| `result can't be cast to Byte` | NPU 不支持 float→uint8 直接转换 | 先 `.cpu()` 再转目标类型 |
| 输出文件写入失败 unknown extension | 工具靠扩展名识别格式 | 输出路径带完整扩展名 |

改张量运算前先在纸上推一遍 broadcast 形状（多加一次 unsqueeze 能把显存申请炸到 TB 级）；
补丁脚本必须带自检断言 + 用 grep 验证落盘；给终端传多行脚本用 base64 单行编码（防自动缩进搞乱）。

---

## 5. 通用踩坑清单（八条军规）

1. **先看提示符再敲命令**：`[C:\~]$` 是自己电脑，`[user@服务器]$` 才是服务器；
2. **pip 装任何包之后都要自检 torch 版本**：依赖解析会悄悄降级 torch，版本错位 = symbol error；
3. **requirements.txt 不能照单全收**：剔除 CUDA 专用包和与本机 Python 不兼容的旧版本；
4. **下载完权重必查体积**：下载器可能报 Success 但留半截文件；
5. **长任务一律 nohup + tail -f**：SSH 会被掐，前台任务随断线陪葬；
6. **共用机器三不原则**：不动别人占的卡、不 kill 别人进程、不装宿主机全局环境；
7. **补丁必须可验证**：正则替换要断言匹配数、改完 grep 确认、每步有验证点；
8. **报错逐层定位别慌**：NPU 适配的报错是一串连环门（设备映射→算子兼容→包缺失→格式细节），每次失败都比上次更接近终点，用最小脚本隔离复现即可。

---

## 6. 常用速查命令

```bash
# 环境
npu-smi info                                          # 卡状态/占用
cat /usr/local/Ascend/driver/version.info             # 驱动版本
python3 -c "import torch,torch_npu; print(torch.__version__, torch.npu.is_available())"

# 容器
docker run -dit --name x --device=/dev/davinci0 ...   # 起容器
docker exec -it x bash                                # 进容器
docker commit x x:v1                                  # 固化环境
docker rm -f x                                        # 删除（挂载目录数据不受影响）

# 下载
modelscope download --model <org>/<name> --local_dir <dir>   # 国内源权重

# 长任务
nohup python run.py ... > run.log 2>&1 &              # 后台
tail -f run.log                                       # 看进度
```

---

## 7. 一页纸决策总结

1. **摸底**：npu-smi 看卡 → version.info 看驱动 → docker images 看库存 → df -h 看磁盘 → ps 看占用；
2. **查配套表**：驱动 → CANN → torch_npu/torch 四个版本号定下来；
3. **选路线**：官方预适配（最优）> MindIE（需 root）> vllm-ascend（LLM）> 通用容器+手工适配（下策）；
4. **起容器**：只透传要用的卡 + 挂载家目录 + 进门自检；
5. **备料**：ModelScope 下权重 + 校验体积 + 甄别式装依赖 + 复查 torch 版本；
6. **跑任务**：nohup 后台 + tail -f 盯进度 + 输出带扩展名；
7. **收尾**：验收产物 → docker commit 固化环境 → 拷贝回本地。

总结
① 看硬件基本信息（npu-smi info）
   → 知道是什么卡、几张、显存多大
   → 决定能跑多大的模型（910B3 64G 和 310P 24G 是两个世界）

② 看驱动/固件信息（version.info）
   → 驱动是底层锁，向下兼容多代固件
   → 它决定"CANN 最高能装到哪个版本"（查官方配套表）

③ 根据"驱动版本 → 配套表"选定 CANN 版本
   → CANN 是整个生态的地基：算子库、图引擎、HCCL 都在这里
   → 版本选错的后果：新算子缺失、老算子行为变化、和上层引擎对不上

④ 根据"配套表里 CANN → 引擎"的对应关系选推理引擎
   → LLM：vllm-ascend（绑定 vLLM 版本）或 MindIE-LLM（绑定 MindIE 版本），
     两者都声明"需要 CANN ≥ x.x"
   → 扩散/多模态：torch_npu（torch_npu 每个 post 版本都标明配哪个 CANN）

⑤ 引擎定了 → 模型的代码/镜像 → 跑
