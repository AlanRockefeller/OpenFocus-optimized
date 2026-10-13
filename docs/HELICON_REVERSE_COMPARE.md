# 用 REA 方法逆向 Helicon Focus 8.1 并与 OpenFocus 对比

> 方法论来自 GitHub 热门项目 [morluto/rea (Reverse Engineer Anything)](https://github.com/morluto/rea)：
> **Decompile → Understand → Recreate**（打开二进制 → 字符串/符号/证据 → 解释 → 复刻）。
> 本次在本机执行了其静态分析层：PE 结构分析、字符串挖掘（10.5 万条）、官方文档考古、运行日志取证。
> 未做任何破解/许可证操作；分析仅用于功能层面的互操作理解。

## 1. 目标二进制画像（binary_overview 等价）

| 项目 | 证据 | 结论 |
|---|---|---|
| 主程序 | `D:\Helicon Focus 8\_HeliconFocus.exe`（22.7 MB），由 412 KB 的 `HeliconFocus.exe` 启动器拉起 | 启动器+实体的双 EXE 结构 |
| 版本 | FileVersion 8.1；运行日志自报 "Helicon Focus 8.1.0 x64"，构建号 `34848bb46a55`，构建于 2022-04-02 | git 提交号可追溯，CI 出品（字符串含 `C:\Jenkins\workspace\heliconfocus64\...`） |
| 技术栈 | 导入表只有 Qt5Core/Gui/Widgets/Network/Svg/Xml + MSVC 运行时 + dxgi + WS2_32 + OpenSSL；日志自报 Qt 5.15.2 / OpenSSL 1.1.1j | C++17 + Qt 5.15.2 + MSVC x64，算法静态链接（无 OpenCV/LibRaw DLL） |
| 加壳判断 | 6 节区标准 MSVC 布局；`.text` 熵 6.46，`.rdata` 高达 16 MB | **未加壳**；`.rdata` 里塞满内嵌资源（多语言 .lng、图标、OpenCL 内核） |
| 签名 | Authenticode 有效，CN="GELIKON SOFT, TOV"（乌克兰哈尔科夫） | Helicon Soft 法律实体，版权 "© 2000 Helicon Soft Ltd" |
| 本机许可 | `%LOCALAPPDATA%\HeliconFocus\HeliconFocus.log`：`Pro license` | 本机为 **Pro 版**；OpenCL 检测到 Intel UHD 770 但未启用 |

内部类名（来自断言/日志宏泄漏的 RTTI 字符串）：`Retouching` / `RetouchingManager` / `RetouchingEditor` / `RetouchingDocumentView` / `RetouchingPaintThread`、`PanoramaEditor` / `PanoramaTask`、`TaskController::compileTask`、`FolderMonitor`、`MultiThreadedTaskRunner`、`LibRawLoader` / `DNGConvertLoader` / `DNGLoader`、`VideoDecoder`（调 ffmpeg）。

## 2. 还原出的功能地图（search_strings + 文档交叉验证）

### 2.1 融合核心
- **三种算法**（官方帮助原文语义）：
  - **Method A（weighted average）**：逐像素按对比度算权重后加权平均。短堆栈、保色彩保对比度。
  - **Method B（depth map）**：逐像素选最清晰的源帧 → 深度图；要求拍摄顺序连续；平滑表面纹理最佳。参数 **Radius**（分析邻域）+ Smoothing（= 深度图平滑，另有 Depth map feathering）。
  - **Method C（pyramid）**：金字塔高低频分离；复杂场景/深堆栈/交叉物体，但会加重对比度和眩光。
  - **Legacy Method B**：HF7 旧实现，默认隐藏，可在 Preferences 打开（快但均匀背景易出伪影）。
- 手工参数只有 Radius/Smoothing 两个主旋钮 + 深度图羽化——极简参数模型。

### 2.2 对齐（Autoadjustments，始终启用、不可换算法）
最大垂直/水平位移（%）、旋转角、缩放差、**亮度均衡**、插值选择（慢速保细节；Lite 宣传材料点名 **Lanczos / Sinc256**）、**手动对焦检测区（ROI）**、自动裁边、用户裁切框、Micro Panorama 单独一组（只平移，不缩放/旋转）。

### 2.3 RAW 管线（对 OpenFocus 是完全缺席的领域）
- 支持 40+ RAW 扩展名（CR2/CR3/NEF/ARW/ORF/RAF/RW2/DNG/…/HEIC）。
- 三种加载器：**内置 dcraw**、**LibRaw**（白平衡自动/相机、高光恢复 Clip/Unclip/Blend/Rebuild、去马赛克 Linear/VNG/PPG/AHD/DCB、色域 raw/sRGB/Adobe/ProPhoto、噪点阈值）、**RAW-in-DNG-out**（调用 Adobe DNG Converter，原生色彩空间处理，输出仍可为 DNG，Pro 专属）。

### 2.4 工作流外围（商业产品的护城河）
- **Retouching 润饰层**：从对齐后的源帧复制（Copy from source）、Clone area / Paint with pattern / Texture only 克隆笔刷、Erase、颜色容差、亮度匹配、源图高亮（Show source map）、F9 快速取源帧、Ctrl+G 网格、多笔刷撤销/重做；"Use another output as a source" 可把不同参数渲染的多个结果互相作为润饰源（Pro）。
- **工程文件 .hproj**：保存润饰历史，跨会话恢复。
- **尘点图 Dust map**：拍一张失焦白场 → 全堆栈去传感器尘点/热像素。
- **批量与拆分**：Batch process（多堆栈逐行设参）+ **Split stacks**（按张数/时间间隔/曝光/HDR 焦点组自动切分）。
- **Folder monitor**：监视文件夹，自动成堆栈并按 `年/月/日/时/分/秒/计数器` 模板命名输出——为延时摄影/自动显微工作流设计。
- **Micro Panorama**（Pro）：显微镜平移拍摄的网格拼接（行/列/重叠度/接缝平滑/拍摄顺序）。
- **Lightroom 插件**：导出→HF 渲染→结果自动回流 LR；`--lightroom-integration-mode`、`--install-lightroom-plugin`、自带 `HeliconFocusDNG.lrtemplate`。
- **Helicon Remote 联动**：一键拉起/下载相机遥控程序（自动对焦括弧拍摄），主程序内嵌大量相机控制字符串。
- **Text/Scale**：标尺库（安装目录 `Scales/*.tif` 15 种）+ 已知尺寸标定 + 元数据插入（日期/光圈/ISO/©/µ）。
- **性能**：OpenCL 硬件加速（Pro，可选设备、失败保险模式、**内嵌多家厂商预编译 OpenCL C 内核**，`--disable-opencl`/`--use-opencl-sources`）；SSE/AVX/AVX-512 运行时分发（`-disable_sse/-disable_avx/-disable_avx512`）；内存缓存配额、并行加载、磁盘缓存、100 帧 Benchmark 联网排名。
- **许可分档**：Lite（16 位、无限堆栈、尘点图、CLI、后台渲染）/ Pro（+RAW-in-DNG-out、润饰、拆分批量、3D 导出、显微全景、动画、OpenCL）/ Premium（+移动端 Helicon Remote）。试用 30 天后限 4 MP + 水印。
- **CLI 模式**：`-silent -save: -j: -dmap -3d -noprogress -i/-o(文件清单) -preferred-output-path -tif:(lzw|u) -mp:(0/1/2) -rp: -sp: -va: -ha: -ra: -ma: -ba: -im: -dmf:`。

## 3. 与 OpenFocus 1.38 的对比

| 维度 | Helicon Focus 8.1 Pro | OpenFocus 1.38 |
|---|---|---|
| 形态 | 闭源商业（Lite $30 / Pro $200 / Premium $350 档），C++/Qt5 静态编译 | 开源免费，Python 3.10+ / PyQt6，GUI + pip `openfocus` CLI |
| 融合算法 | 3 种经典（加权平均 / 深度图 / 金字塔）+ legacy B | **5 种**：Guided Filter、DCT、DTCWT、GFG-FGF、**StackMFF-V4 神经网络**（Torch，CUDA/MPS/CPU） |
| 对齐 | 隐式内置，仅全局参数（位移/旋转/缩放上限、亮度均衡、插值） | 显式可选 **ECC 或 SIFT+RANSAC 单应**（`-a`），可关闭 |
| 参数模型 | Radius/Smoothing 两旋钮 | kernel_size（奇数校验）、AI batch、线程数、tile 三参数 |
| 超大图 | 内存缓存配额 + 磁盘缓存 | **分块融合引擎**（1024 块/256 重叠，自动触发）+ AI 显存预算 |
| GPU | **OpenCL 加速经典算法**（Pro，内嵌多厂商内核） | 经典算法**刻意 CPU-only**；GPU 只给神经方法 |
| RAW | 40+ 格式、3 加载器、专业 RAW 开发参数 | ❌ 无 RAW 解码（依赖外部转 TIFF/JPEG） |
| 深度图 | 灰度导出 + 羽化 | ✅ 5 种清晰度度量 + turbo 等**彩色映射** |
| 润饰笔刷 | ✅ 完整 PS 式润饰（源帧复制/克隆/纹理） | ❌ 无 |
| 尘点图 | ✅ | ❌ 无 |
| 拼接/3D | 显微全景拼接、3D 模型导出、红蓝立体 | ❌ 无 |
| 批量 | 批处理 + 智能拆堆栈 + 文件夹监视自动出图 | 批量多文件夹任务（无拆分/监视） |
| 工程文件 | .hproj（含润饰历史） | 项目文件 + **崩溃自动恢复**（`%TEMP%` 诊断 + `--restore-session`） |
| 标尺 | 标尺库 + 手工标定 | 发布级标尺（µm/px 标定，程序化） |
| 输出 | JPEG/TIFF(LZW)/DNG/PNG、图层导出、动画(HTML+ffmpeg)、发布到网/Facebook | PNG/JPEG/TIFF/**多页 TIFF Z 栈**、GIF；16 位端到端保证 |
| 生态 | Lightroom 往返、Helicon Remote 相机联动、联网跑分 | GitHub 开源、PyPI、CI、每日后台自更新 |
| CLI | 单字母短参数，面向静默集成 | `--json/--json-progress/--resume/--dry-run/--config`，面向自动化/CI |
| i18n | 内嵌 .lng（含中文 UI）+ 6 语帮助 | en/zh/ja/es 四语，测试强制键位一致 |

### 一句话结论
Helicon 的核心竞争力不在融合算法本身（三家方法都是 2000 年代经典思路），而在**围绕显微/摄影工作流的整条链**：RAW 开发 → 自动对齐 → 融合 → **润饰笔刷** → 标尺/尘点图 → 批量/监视 → Lightroom/Remote/3D 生态，以及 OpenCL 性能层。OpenFocus 在算法多样性（含神经方法）、配准可控性、分块引擎、开源可扩展性和现代 CLI 上反超，但在 RAW、润饰、拼接/3D、相机生态上是空白。

### REA 式"Recreate"建议（OpenFocus 可吸收的清单，按性价比排序）
1. **尘点图**——实现成本最低、显微用户痛点最强（一张失焦白场做修复源）。
2. **深度图羽化参数**——`core/depth_map.py` 加 smoothing，半天工作量。
3. **加权平均法（Method A）**——补齐短堆栈/保色彩场景，比 DCT 还简单。
4. **源帧复制式润饰**——OpenFocus 已有对齐后的帧序列，缺的只是笔刷 UI 层。
5. **文件夹监视自动出图**——`BatchManager` 天然适合扩展成 watchdog。
6. OpenCL/AVX 分发——与项目"经典算法 CPU-only"原则冲突，可不做。

---
### 证据文件（本次逆向的 artifacts）
- `F:\Working\helicon-rea\strings_ascii.txt` — 105,614 条 ASCII 字符串
- `F:\Working\helicon-rea\strings_utf16.txt` — 450 条宽字符串
- `F:\Working\helicon-rea\dll_refs.txt` — 导入 DLL 清单
- `F:\Working\helicon-rea\help_english.txt` — 官方英文帮助全文（45 KB 纯文本）
- 运行日志：`C:\Users\dell\AppData\Local\HeliconFocus\HeliconFocus.log`
- 目标：`D:\Helicon Focus 8\_HeliconFocus.exe`（8.1.0 x64，sha 无需记录，签名有效）
