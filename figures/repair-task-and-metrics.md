# 修补任务生成与四个质量指标

下图把论文的“分割掩膜到执行轨迹”和“执行后像素评价”分开，避免把轨迹生成误解成完整的在线路径规划或闭环控制。

```mermaid
flowchart LR
    image[工业相机图像] --> seg[HrSegNet-B32-AD<br/>二值裂缝掩膜]
    seg --> roi[操作员确认 + ROI]
    roi --> prep[反相 + 双边滤波]
    prep --> thin[Zhang-Suen 细化<br/>单像素骨架]
    thin --> geom[宽度/长度/面积<br/>像素到毫米标定]
    thin --> spline[B-spline 拟合<br/>连续中心线]
    geom --> speed[按 v(s)=Q/(k*w(s)*h(s))<br/>调节针头速度]
    spline --> transform[手眼标定<br/>像素坐标 -> 机器人基座坐标]
    speed --> command[XY 分解 + 步进频率/步数]
    transform --> command
    command --> inject[3-DOF 机构沿轨迹注胶]
    inject --> post[修补后胶体掩膜]
    seg --> compare[与原裂缝掩膜对齐]
    post --> compare
    compare --> metrics[CR / UR / OR(OFR) / CUI]

    classDef perception fill:#E3F2FD,stroke:#1565C0,color:#111;
    classDef planning fill:#F3E5F5,stroke:#6A1B9A,color:#111;
    classDef execution fill:#FFF3E0,stroke:#EF6C00,color:#111;
    classDef eval fill:#E8F5E9,stroke:#2E7D32,color:#111;
    class image,seg,roi,prep,thin,geom perception;
    class spline,speed,transform,command planning;
    class inject,post execution;
    class compare,metrics eval;
```

