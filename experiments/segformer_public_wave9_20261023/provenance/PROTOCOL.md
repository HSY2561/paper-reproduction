# Public comparison protocol
- RTX 5060 / FP32 / Python C:\venv_bisenet\Scripts\python.exe
- Data: C:\Users\user\algolab_data; test is accessed only after frozen validation checkpoints exist.
- Official PaddleSeg source snapshot: D:\论文复现\src\PaddleSeg-2.10.0
- Metrics: global confusion-matrix mIoU, foreground crack IoU, precision, recall, F1; zero-extended 3x3 inner-boundary and exact Euclidean <=2 px matching.
- Checkpoints are reused only after validation selection; no public model is modified.
