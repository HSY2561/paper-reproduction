# W11 执行跟踪

实际更新时间：2026-10-25T20:04:11.055222+08:00；总15项＝12四格＋3匹配R0。当前完整训练及独立验证完成2/15；尚不能判断三seed互补。

实际心跳：W11_E2_s42_train，6100/18000（心跳可能落后于实时终端）。后处理等待训练真实退出。

| Run | 组别 | seed | 状态 | 真实train exit |
| --- | --- | ---: | --- | --- |
| W11_E0_s42 | E0 | 42 | complete | 0 |
| W11_E1_s42 | E1 | 42 | complete | 0 |
| W11_E2_s42 | E2 | 42 | training | 未知/未结束 |
| W11_E3_s42 | E3 | 42 | pending | 未知/未结束 |
| W11_E0_s43 | E0 | 43 | pending | 未知/未结束 |
| W11_E1_s43 | E1 | 43 | pending | 未知/未结束 |
| W11_E2_s43 | E2 | 43 | pending | 未知/未结束 |
| W11_E3_s43 | E3 | 43 | pending | 未知/未结束 |
| W11_E0_s44 | E0 | 44 | pending | 未知/未结束 |
| W11_E1_s44 | E1 | 44 | pending | 未知/未结束 |
| W11_E2_s44 | E2 | 44 | pending | 未知/未结束 |
| W11_E3_s44 | E3 | 44 | pending | 未知/未结束 |
| W11_R0_s42 | R0 | 42 | pending | 未知/未结束 |
| W11_R0_s43 | R0 | 43 | pending | 未知/未结束 |
| W11_R0_s44 | R0 | 44 | pending | 未知/未结束 |

原始证据以results状态/真实exit和checkpoints选择记录为准。本页只读聚合，不更改训练汇总JSON，不读取test/自建集，不以smoke代替正式结果。
