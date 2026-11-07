# W11 E0–E3 执行跟踪

更新时间：2026-10-25T19:02:49.340795+08:00。12个核心四格＋3个匹配R0，总15次18k；原预登记12组计划不覆盖，追加依据见EXECUTION_ADDENDUM。

实际阶段：formal_training。最新心跳：W11_E0_s42_train，iter 1800/18000。心跳可能落后于终端；下表exit未知者保持未知。

| Run | 组 | seed | 状态 | 真实train exit |
| --- | --- | ---: | --- | --- |
| W11_E0_s42 | E0 | 42 | training | 未知/未结束 |
| W11_E1_s42 | E1 | 42 | pending | 未知/未结束 |
| W11_E2_s42 | E2 | 42 | pending | 未知/未结束 |
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

正常运行不重复启动。操作状态以results/workflow_state.json、heartbeat.json及真实exit为准；原source/init/config已冻结，禁止在训练期间修改。只用train/val。
