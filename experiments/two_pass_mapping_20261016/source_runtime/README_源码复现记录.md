# 作者源码复现记录（2026-10-16）

## 1. 源码身份和来源

- 论文：Veeraraghavan et al., *Complete and Near-Optimal Robotic Crack Coverage and Filling in Civil Infrastructure*, IEEE T-RO 2024, DOI `10.1109/TRO.2024.3392077`。
- 作者仓库：<https://github.com/Binghamton-ACSR-Lab/Crack-Filling-Robot>。
- 本次核验上游 main：`d044d083fd9fb21296f9b8fb5d55c6b42145bf92`，提交日期 2025-10-05。
- 已有本地源码：`D:\论文复现\references\Crack-Filling-Robot`。
- README 与 Software 下全部 36 个文件，经 Git blob SHA-1 与官方树逐一核验，36/36 相同，见 `official_source_verification.json`。这些源码已经存在，无需重复下载整个 171 MB 量级的仓库。
- 本地此目录没有独立 `.git`，`git -C` 会回溯到论文工程仓库；因此不能把论文工程的 HEAD 当作作者版本。上游身份由 GitHub API 与逐文件核验确认。
- 官方 API 返回 license=null，根目录未找到 LICENSE。可公开获取不等于任意再发布许可；当前仅在本地保留作者归属并做复现，不在本任务对外分发作者源码。

## 2. 实际运行环境

- MATLAB：`C:\Program Files\MATLAB\R2025b\bin\matlab.exe`。
- 版本回执：`25.2.0.2998904 (R2025b)`。
- Image Processing、Mapping、Statistics toolbox license test 均返回 1。
- 完整 SCC 中的 `intlinprog` 实际运行成功。
- MATLAB 探针：exit code=0，stderr 为空；原始 stdout/stderr、启动命令和结果见 `matlab_probe.*` / `matlab_probe_launch.json` / `matlab_probe_result.json`。
- 图形后处理：既有 `C:\Users\user\算法实验\venv_bisenet\Scripts\python.exe`，scipy + matplotlib 3.11.2。

## 3. 本次运行边界与实际结果

| 项目 | 实际输入 | 调用 | 真实结果 | 边界 |
|---|---|---|---|---|
| 完整 SCC | 官方 `myCrack8_100_1.png` 人工地图 | 作者 `SCC.m` | exit=0；stderr空；`SCC_COMPLETE`；158个路径点、39个图节点、32条边 | 已知地图仿真，未执行真实扫描或机器人控制 |
| 弓字形模块 | 3048×2896像素矩形 | 未改动的 `BoustrophedonPath` | exit=0；77个路点；传感半径342px；行间距483.661px | 人工工作区/作者像素尺度，未做差速转弯约束或实际轨迹跟踪 |
| 真实 SegFormer mask 接口 | 既有 P0 的360×640预测mask | 未改动的 `Image_planning_func_SCC(mask)` | exit=0；2个节点、1条边 | 复用此前真实预测；本轮没有重新神经网络推理，也不是多帧拼图 |

完整 SCC 的标准输出原样为：

```text
    2.3976  135.7403

   95.9561  100.0000  138.1379  164.5500  521.0750

SCC_COMPLETE
```

作者 `res` 字段依次为：缓冲路径面积/工作区面积的百分比、裂缝密度档、代码记录的计算耗时秒数、路径长度英尺、累计扫描面积平方英尺。它不是本项目的地图精度/机器人修复率，也不能直接替代实际传感覆盖测量。作者假设 2 mm/px，不是我们的相机标定结果。

完整 SCC 运行从 `00:51:50+08:00` 到 `00:55:00+08:00`（含 MATLAB 启动/退出）；程序内部记录计算耗时约138.14秒。完整回执在 `scc_exit.json`，原始输出为 `scc.stdout.log`、`scc.stderr.log`。

## 4. 为可运行做了什么

仅在 `author_software` 隔离副本中：

1. 复制作者 Software 和官方示例图；原始仓库没有修改。
2. 创建作者脚本要求的 `Crack Maps/Uniform` 路径。本地库原名称是 `CrackMaps`，这是路径兼容处理。
3. 创建 `Results/GIF` 输出目录。
4. `SCC.m` 唯一改动是 `worki=true` 改为 `worki=false`，关闭大量逐帧动画导出。没有修改规划逻辑。
5. `run_modules.m` 是调用/记录/绘图的实验入口，直接调用作者函数；`render_and_summarize.py` 只读取 MATLAB 结果绘图，不重新计算路径。

具体记录：`harness_changes.json`。原始来源校验：`official_source_verification.json`。输出和副本文件哈希：`artifact_sha256.json`。

## 5. 可查看的结果

- `scc_official_simulation.png` / `.svg`：作者完整 SCC 路线与原始人工地图。
- `scc_result.mat`：真实 MATLAB 工作区输出。
- `scc_path_px.csv`：158个底盘覆盖路径点，x/y像素。
- `scc_nodes_row_col.csv` / `scc_edges_1based.csv`：图结构，节点坐标为行/列，边索引从1开始。
- `boustrophedon_official_module.png`：作者弓字形矩形覆盖模块。
- `boustrophedon_path_px.csv`：弓字形77路点。
- `segformer_author_graph_smoke.png`：真实预测mask、作者骨架处理、图抽象。
- `segformer_raw_skeleton.png`、`segformer_author_pruned_skeleton.png`：剪枝前后骨架。
- `segformer_graph_nodes_row_col.csv`、`segformer_graph_edges.csv`、`segformer_author_graph.mat`：真实mask接口的图输出。
- `run_summary.json`：结构化汇总。

## 6. 本次接口测试发现的关键问题

作者图处理内硬编码：`2 mm/px`、机器人直径48英寸、作业直径7英寸、传感直径54英寸。这导致 `a=44px`，并将该数用于 `bwmorph(...,'spur',a)` 和短段删除。

在本项目 P0 mask 上：前景13354像素；初始骨架526像素；作者剪枝后429像素，减少97像素（18.44%）。这不是人工判定的错误率，但说明原参数会显著改变端点/短支路。当前缺少真实标定，不应把作者44px阈值搬到我们的数据。

作者输出图以两个端点的一条直边描述该裂缝，用于底盘作业覆盖的几何抽象。这条直线并不严格贴合原始曲线，因此不能直接发送给喷嘴。底盘粗定位图和局部喷嘴中心线必须分开保存；局部精修仍需高分辨率骨架/原始曲线。

## 7. 本文可复用与尚缺的接口

可复用：

- `BoustrophedonPath(cell,orgcell,reebEdge,bp_gap,dir,init,wall_fol,known,fl_see)`：输入多边形、行间距和起点，输出顺序坐标；依赖若干global变量。实际代码相邻扫描列间距是 `2*bp_gap`。
- `Image_planning_func_SCC(mask)`：可以直接接收二值矩阵，跳过字符串路径的固定读图部分；输出行/列节点与边列表。
- `MCD`、`Reeb`、`ReebPath`、`ChinesePostman`：作为后续复杂工作区/路径覆盖参考，现阶段矩形场地无需全部引入。
- `OnlineSCC`：本轮只审阅未执行。它在已知全图中按传感范围截取观测以模拟未知发现，不能直接替代相机拼接建图。

尚缺：相机标定、同步图像和底盘位姿、世界坐标投影与融合、配准质量检查、真实未知地图接口、差速底盘控制、局部复扫、三轴执行与喷嘴闭环。这些没有因 MATLAB 示例成功而自动完成。作者仓库提供的是规划 MATLAB 代码、资料和仿真地图，没有在这次核对的树中发现相机SLAM或机器人实时驱动源码。

扫描必须使用相机有效地面覆盖宽度确定行间距，不能套用末端覆盖间距 `2a`；本轮路点仍以作者像素坐标呈现，尚不构成真实场地的可执行米制轨迹。

## 8. 重现命令

MATLAB模块（PowerShell）：

```powershell
& 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe' -batch "addpath('D:/论文复现/experiments/two_pass_mapping_20261016/source_runtime'); run_modules;"
```

SCC（输出写在本次隔离目录）：

```powershell
& 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe' -batch "cd('D:/论文复现/experiments/two_pass_mapping_20261016/source_runtime/author_software'); set(groot,'defaultFigureVisible','off'); SCC; save('../scc_result.mat','PathEdge','res','node','edgeList','s','a','pathLength'); disp('SCC_COMPLETE');"
```

实际启动的完整参数、时间、PID、stdout、stderr及退出码保存在 `*_launch.json` 与 `*_exit.json`；不需要从文字报告推断命令是否执行。
