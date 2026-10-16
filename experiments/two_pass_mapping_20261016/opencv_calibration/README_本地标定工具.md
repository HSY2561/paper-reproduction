# OpenCV官方标定工具，本地已验通

原始源码来自OpenCV 4.11.0；源码URL和文件哈希见 `source_manifest.json`，保留LICENSE。没有自建标定算法。

运行环境：`C:\Users\user\算法实验\venv_bisenet\Scripts\python.exe`，OpenCV 4.11.0。该环境为无GUI版，使用官方 `calibrate.main()` 可避免脚本末尾无必要的 `destroyAllWindows()` 报错。计算主体与官方源文件完全一致。

复现本轮软件验通：在本目录执行以下PowerShell命令。只处理官方样图，不使用你的相机。

```powershell
$sampleImages = Get-ChildItem -LiteralPath .\data -Filter 'left??.jpg' | Sort-Object Name | ForEach-Object { 'data/' + $_.Name }
& 'C:\Users\user\算法实验\venv_bisenet\Scripts\python.exe' -c 'import calibrate; calibrate.main()' -w 9 -h 6 -t chessboard --square_size=1 --marker_size=0.5 --threads=1 --debug=outputs_headless @sampleImages
```

真实回执：`headless_run_receipt.json`，exit=0；13/13图像识别，RMS=0.1962351132525735 px。所有结果只属于OpenCV示例相机，格长单位是1个棋盘格，不能填入机器人标定文件。普通脚本入口的GUI清理失败证据保留于 `run_receipt.json` 和 `stderr.log`。

可打印标定板：`checkerboard_10x7_squares_20mm_A4.svg`，横向A4（297×210 mm），10×7方格，9×6内角点，20 mm标称格长。用官方 `gen_pattern.py` 生成，回执 `pattern_absolute_run_receipt.json`。使用SVG按100%/实际大小打印；PNG仅作预览。打印后量横纵连续若干格，记录实际值；纸张贴硬板、防翘曲。若工作高度下角点太小，需按相机视场选择更大的板，不能靠图像放大补足分辨率。

调用用户真实照片前，要确认实际内角点数与格长。`--square_size` 使用实测单位，若用米则20 mm约为0.02；必须以实测为准。输出目录与原图分离。Windows下优先在数据工作目录内传英文相对文件名，避免旧版OpenCV对中文文件路径的支持问题。官方示例会打印K和畸变，后续真实标定还需额外记录机器可读参数、独立验证和标定版本；示例本身不是完整机器人标定交付。
