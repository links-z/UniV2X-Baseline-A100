# UniV2X Occupancy 数据流追踪诊断报告

**时间**：2026-09-19（2026-09-19 修正）
**分支**：agent-fusion-improved
**任务**：定位完整的 Occupancy 数据流（第二阶段任务）

> **修正记录（2026-09-19）**：本报告初版有 7 处技术表述与实际代码不符，已逐条对照 `occ_head.py` / `apis/test.py` / `univ2x_coop_e2e.py` 改正。修正项在文中以 `【修正】` 标注，并在第九节汇总。核心结论"官方 Occupancy 融合最终在二值空间做 max（等价于逻辑 OR）"不变。

---

## 一、Occupancy 评估两个范围的确认

### 1.1 官方评估范围定义

**代码位置**：`projects/mmdet3d_plugin/univ2x/apis/test.py:63-64`

```python
EVALUATION_RANGES = {'30x30': (70, 130),
                    '100x100': (0, 200)}
```

**含义**：
- 两个评价范围都定义在原始 200×200 的 BEV 网格中
- **Range1 (30×30)**：栅格切片 `[70:130, 70:130]` → 60×60 的中心区域
- **Range2 (100×100)**：栅格切片 `[0:200, 0:200]` → 整个 200×200 区域
- 调用处 `test.py:117-120`：IoU / PQ 都在 `[..., limits, limits]` 上计算，**时间维 t 完整参与**（见第 4.3 节）

### 1.2 最新评估结果（9月18日）

```
Occ-flow Val Results:
━━━━━━━━━━━━━━━━━━━━━━
IoU:   22.6 (Range1) & 25.9 (Range2)
PQ:    2.0 (Range1) & 0.1 (Range2)
SQ:    63.1 (Range1) & 60.2 (Range2)
RQ:    3.2 (Range1) & 0.1 (Range2)
━━━━━━━━━━━━━━━━━━━━━━
样本数：549/675 (81.3%)
```

**观察**：Range2 (全局) 的 PQ/RQ 远低于 Range1 (局部)，说明外围区域预测质量较差。

---

## 二、完整 Occupancy 数据流路径

### 2.1 Occupancy 张量命名约定

| 符号 | 含义 | Shape | 数值范围 | 代码位置 |
|------|------|-------|---------|--------|
| $P_{v}^{\text{logit}}$ | 车端原始 Occupancy | [b, q, t, h, w] | **logits（已确认）** | forward_test L463/465 |
| $P_{v}^{\text{sig}}$ | 车端 sigmoid 后 | [b, q, t, h, w] | [0, 1] | forward_test L468 |
| $P_{seg}$ | 沿 query 维取 max（q 消失） | [b, t, h, w] | [0, 1] float | forward_test L477 |
| $P_{\inf}$ | 路侧传来的 Occupancy | [b, t, h, w] | **[0,1] float 概率** | forward_test L485 |
| $P_{\inf}^{\text{align}}$ | 空间对齐后 | [b, t, h, w] | [0, 1] float | occ_prob_fusion L539 |
| $P_{f}$ | **最终融合输出** | [b, t, h, w] | **0/1 long** | occ_prob_fusion L551 |
| $Y$ | Ground Truth | [b, t, h, w] | 0/1 | forward_test L454-455 |

【修正②】$P_{\inf}$ 是 **float 概率**，不是 0/1（见 L504 传输的是阈值化*之前*的 `pred_seg_scores[0]`）。
【修正③】$P_{v}^{\text{logit}}$ 确认是 logits，L468 显式 `.sigmoid()`，无需再猜。

### 2.2 数据流图

```
┌──────────────────────────────────────────────────────────────┐
│                  forward_test() 入口 (occ_head.py L438)       │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  L463: _, pred_ins_logits = self(bev_feat, ins_query)        │
│  输出：pred_ins_logits [b, q, t=5, h=200, w=200]             │
│  格式：logits（已确认，因 L468 紧接 sigmoid）                 │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  L468: pred_ins_sigmoid = pred_ins_logits.sigmoid()          │
│  输出：[b, q, t=5, h=200, w=200]，概率 ∈ [0, 1]              │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  L477: pred_seg_scores = pred_ins_sigmoid.max(1)[0]          │
│  操作：沿 query 维 (dim=1) 取最大值 → q 维消失               │
│  输出：pred_seg_scores [b, t=5, h=200, w=200]，float [0,1]   │
│  ★ 融合永远发生在这个「无 q 维」的 float 概率图上            │
└────────────────────────────┬─────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │ is_ego_agent && │
                    │ is_cooperation &&│
                    │ other_agent?    │
                    └────────┬────────┘
        ┌────────────────────┴────────────────────┐
       否                                        是
        │                                         ▼
        │          ┌──────────────────────────────────────────────┐
        │          │  L485: other_agent_occ_data                  │
        │          │   = ...['occ']['univ2x_occ_prob_data']       │
        │          │  【修正②】格式：float 概率（0..1），非 0/1   │
        │          └──────────────────────┬───────────────────────┘
        │                                 ▼
        │          ┌──────────────────────────────────────────────┐
        │          │  L487: occ_prob_fusion(pred_seg_scores,      │
        │          │        other_agent_occ_data, ego2other_rt)   │
        │          └──────────────────────┬───────────────────────┘
        │                                 ▼
        │          ┌──────────────────────────────────────────────┐
        │          │  【融合内部】occ_prob_fusion() L539-546      │
        │          │  (1) 空间对齐（float 概率上做）：             │
        │          │      new_inf_occ = grid_sample(inf_occ,grid) │
        │          │  (2) 各自阈值化 → 二值：                      │
        │          │      veh_occ_log = (veh_occ  > 0.1).long()   │
        │          │      inf_occ_log = (new_inf_occ> 0.1).long() │
        │          │      【修正①】阈值是 0.1（config 覆盖），非0.5│
        │          │  (3) 二值空间 Max = 逻辑 OR：                 │
        │          │      max(veh_occ_log, inf_occ_log)           │
        │          │  (4) 输出 0/1 long：fused_occ [b,5,200,200]  │
        │          └──────────────────────┬───────────────────────┘
        │                                 ▼
        │          ┌──────────────────────────────────────────────┐
        │          │  L488: pred_seg_scores = new_pred_seg_scores │
        │          │  ⚠️ float 概率被 0/1 long 覆盖（仅 ego 侧）  │
        │          └──────────────────────┬───────────────────────┘
        └────────────────────┬────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  L490: seg_out = (pred_seg_scores > 0.1).long().unsqueeze(2) │
│  【修正①】阈值 0.1。融合后若已是 0/1，则 1>0.1→1, 0>0.1→0   │
│  输出：seg_out [b, t=5, 1, h=200, w=200]，0/1 long           │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  L503-504 (仅 not is_ego_agent，即路侧侧输出)：              │
│   out_dict['univ2x_occ_prob_data'] = pred_seg_scores[0]     │
│  【修正②】此处 pred_seg_scores 未经阈值化 → 传出 float 概率  │
│           （路侧不进入 L481 的 ego 融合分支）                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 三、关键发现

### 🔴 核心结论（不变）：ego 侧融合最终在二值空间做 max = 逻辑 OR

**证据**（occ_prob_fusion L541-546）：

```python
# 各自阈值化为二值（阈值 = 0.1，来自 config 覆盖）
veh_occ_log = (veh_occ > self.test_seg_thresh).long()      # L541
inf_occ_log = (new_inf_occ > self.test_seg_thresh).long()  # L542

# 二值空间 Max（等价于逐像素 OR）
for i in range(self.n_future+1):
    max_values, _ = torch.max(torch.stack([veh_occ_log[:,i],
                                          inf_occ_log[:,i]]),
                             dim=0)  # L546  取值 ∈ {0,1}

fused_occ = torch.stack(fused_occ, dim=2).squeeze(1)  # L551  dtype=long
```

**语义**：`fused = (veh_prob>0.1) OR (align(inf_prob)>0.1)`。

### 问题分析

| 问题 | 影响 | 严重性 |
|------|------|--------|
| **融合前先阈值化 → 二值 OR** | 融合环节丢弃概率信息，无法做加权/概率融合 | 🔴 高 |
| **OR 只增不减** | OR 只能增加占据像素、永不删除 → recall↑、precision 可能↓ | 🔴 高 |
| **ego 融合输出被 0/1 覆盖** | 融合后的 `pred_seg_scores` 不再是概率（仅 ego 分支） | 🟡 中 |
| **评估在二值 seg_out 上** | IoU/PQ 本就基于二值；概率校准（Brier/ECE）需在阈值化前另取 | 🟡 中 |

> 注意：**传输链路仍是 float 概率**（L504）。丢失概率只发生在 ego 侧 `occ_prob_fusion` 内部的阈值化那一步。这对 RA-OccFusion 是有利的——上游概率信息仍在，改造点集中在 `occ_prob_fusion`。

---

## 四、Occupancy 张量形状确认

### 4.1 OccFormer 输出

| 张量 | Shape | 说明 |
|------|-------|------|
| `pred_ins_logits` | [b, q, t, h, w] | q=query 数，t=5（当前+4步未来） |
| `pred_seg_scores`（融合输入） | [b, t=5, 200, 200] | **q 维已被 L477 max 掉** |

### 4.2 维度映射

```
[b]atch
  └─ 推理时通常为 1（逐场景评估）

[q]uery
  ├─ pred_ins_logits 的 query 维，来自 track/instance query
  ├─ 每个 query 一张 occupancy map
  └─ 【修正⑤】L477 max(1)[0] 在融合前就消掉 q →
     融合张量根本没有 q 维。运行时 q 的具体数值取决于当帧
     track query 数（no_query 分支见 detectors/univ2x_e2e.py:355），
     静态代码无法断定「q=1」，需实际打印确认。

[t]ime
  ├─ t=5：当前帧 + 4 步未来（n_future=4）
  └─ 【修正⑦】评估用全部 5 个时间步（见 4.3），非「仅当前帧」

[h], [w] = 200×200（两套网格，见 4.4）
```

### 4.3 评估的时间维度【修正⑦】

- `seg_gt = gt_segmentation[:, :1+n_future]` → **5 帧**（forward_test L454）
- `apis/test.py:117-118`：`iou_metrics[key](seg_out[...], seg_gt[...])` 传入完整 t=5
- 结论：IoU / PQ 在 **5 个时间步**上聚合，不是只算当前帧。

### 4.4 两套网格的区分【修正④】

| 网格 | 定义来源 | 范围 | 分辨率 |
|------|---------|------|--------|
| **Occ GT / 评估网格** | `occflow_grid_conf` xbound/ybound = `[-50, 50, 0.5]` | 100m × 100m | **0.5 m** |
| **BEV 特征 / pc_range 网格** | `point_cloud_range = [-51.2, 51.2]` | 102.4m × 102.4m | **0.512 m** |

- 初版把两者都写成 0.512m，是混淆。
- 附带隐患：`occ_prob_fusion` 的 `grid_sample` 用的是 **pc_range（102.4m）** 网格坐标，而 GT/eval 在 **100m** 网格上——两套网格并不严格重合，空间对齐存在细微不一致，值得在诊断中单独核验。

---

## 五、Max Fusion 代码定位

### 5.1 精确位置

| 项目 | 值 |
|------|-----|
| **文件** | `projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py` |
| **类 / 方法** | `OccHead.occ_prob_fusion()` |
| **行号** | 508-553（核心 L541-546） |

### 5.2 关键代码片段

```python
def occ_prob_fusion(self, veh_occ, inf_occ, veh2inf_rt):
    """
    veh_occ: [1, 5, 200, 200]  车端 float 概率
    inf_occ: [1, 5, 200, 200]  路侧 float 概率（经传输链路传来）
    veh2inf_rt: [1, 4, 4] → 内部取 [0,1,3] 变为 3x3
    返回:
        fused_occ:   [1, 5, 200, 200]  0/1 long（OR 结果）
        inf_occ_log: [1, 5, 200, 200]  对齐后路侧二值（L487 接收后未再使用）
    """
    new_inf_occ = F.grid_sample(inf_occ, bev_grid, align_corners=True)  # L539

    veh_occ_log = (veh_occ > self.test_seg_thresh).long()      # L541  thr=0.1
    inf_occ_log = (new_inf_occ > self.test_seg_thresh).long()  # L542  thr=0.1

    fused_occ = []
    for i in range(self.n_future+1):        # i in [0..4]
        max_values, _ = torch.max(torch.stack([veh_occ_log[:,i],
                                               inf_occ_log[:,i]]), dim=0)
        fused_occ.append(max_values.unsqueeze(1))

    fused_occ = torch.stack(fused_occ, dim=2).squeeze(1)  # [1,5,200,200] long
    return fused_occ, inf_occ_log
```

### 5.3 Fusion 输入来源

```python
# forward_test L481-488（仅 ego 侧进入）
if self.is_ego_agent and self.is_cooperation and other_agent_results:
    for other_agent_name, other_agent_result in other_agent_results.items():
        other_agent_occ_data = other_agent_results[other_agent_name][0]['occ']['univ2x_occ_prob_data']
        # 【修正②】这是路侧输出的 float 概率（L504 未阈值化），非 0/1
        other_agent_occ_data = torch.stack([other_agent_occ_data], dim=0)
        new_pred_seg_scores, new_inf_occ = self.occ_prob_fusion(
            pred_seg_scores,        # [1,5,200,200] float
            other_agent_occ_data,   # [1,5,200,200] float
            other_agent_results[other_agent_name][0]['ego2other_rt'])
        pred_seg_scores = new_pred_seg_scores  # 被 0/1 long 覆盖
```

---

## 六、关键参数确认

### 6.1 网络配置参数

| 参数 | 值 | 来源 |
|------|-----|------|
| `test_seg_thresh`（默认签名） | 0.5 | occ_head.py L60 |
| `test_seg_thresh`（**实际生效**） | **0.1** | `univ2x_coop_e2e.py:444`（config 覆盖）【修正①】 |
| `bev_h` / `bev_w` | 200 / 200 | config L28-29 |
| `n_future` | 4 | config `occ_n_future=4` L85 |
| 车端 `pc_range` | [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] | config L16 |
| 路侧 `inf_pc_range` | [0, -51.2, -5.0, 102.4, 51.2, 3.0] | config L13 |
| Occ GT 网格 | xbound/ybound `[-50,50,0.5]` → 0.5m | `occflow_grid_conf` L36-40 |

### 6.2 空间变换关系

```
车端坐标系：X [-51.2, 51.2]，Y [-51.2, 51.2]（pc_range，102.4m）
路侧坐标系：X [0, 102.4]，  Y [-51.2, 51.2]（inf_pc_range，不对称！）
变换：ego2other_rt [1,4,4]，融合内部取 [0,1,3] 行列 → 3x3 平面变换
```

---

## 七、修正前后对照（供快速复核）

| # | 初版表述 | 判定 | 实际代码 |
|---|---------|------|---------|
| ① | 阈值 = 0.5 | ❌ 错 | config 覆盖为 **0.1**（L444 vs 签名默认 L60） |
| ② | 路侧传 0/1 | ❌ 错 | 传 **float 概率**（L504 阈值化前的 pred_seg_scores[0]） |
| ③ | logits 是否 sigmoid 还需验证 | ❌ 多余 | 已明确：logits → `.sigmoid()`（L468） |
| ④ | 分辨率 = 0.512m | ⚠️ 混淆 | Occ GT/eval **0.5m**；BEV pc_range **0.512m**，两套网格 |
| ⑤ | q 是检测 query 数（对融合张量） | ❌ 不成立 | q 在 L477 融合前被 max 掉；运行时数值需打印，不能断言=1 |
| ⑥ | 用 Max Selection Accuracy | ❌ 错配 | 官方是二值 **OR**，非「二选一取高置信」，应改用 OR 误差拆解 |
| ⑦ | 仅评估当前帧 GT | ❌ 错 | IoU/PQ 用全部 **5 个时间步**（L454 + test.py:117-118） |

**唯一保持不变的核心结论**：ego 侧最终融合 = `OR(veh>0.1, align(inf)>0.1)`，二值空间。

---

## 八、第三阶段诊断设计（据修正后重写）

### 8.1 正确的数学定义（钉死）

```
传输：inf 侧输出 float 概率 P_inf ∈ [0,1]   (occ_head L504)
对齐：P_inf^align = grid_sample(P_inf, ego→inf 网格)   (L539, 用 pc_range 102.4m 网格)
融合：P_f = OR( P_v > 0.1 , P_inf^align > 0.1 )         (L541-546, 二值)
评估：IoU/PQ( P_f 二值, GT )，t=5，两个空间范围（0.5m 网格）
```

### 8.2 零训练诊断项（Go/No-Go 依据）

1. **四基线对比**：Ego-only / Infra-only / Mean（概率均值后阈值）/ 官方 OR
   （Mean 与 OR 的差距，直接说明"保留概率再融合"是否有空间）
2. **OR 误差拆解**（替代原"Max Selection Accuracy"）——逐像素分四类：
   - `正确补充`：veh=0,inf=1,GT=1（OR 收益）
   - `假阳叠加`：veh=0,inf=1,GT=0（OR 代价，precision 损失来源）
   - `双错`：veh=0,inf=0,GT=1（OR 无能为力）
   - `双对/冗余`：veh=1,inf=1
3. **冲突比例**：`veh_bin != inf_bin` 的像素占比（可靠性有无信号）
4. **概率信息可用性**：在阈值化*前*统计 P_v / P_inf 的分布与相关性

### 8.3 待实测确认（需打印，不臆断）

```
□ pred_ins_sigmoid.shape → 确认运行时 q 值
□ pred_seg_scores 数值范围（阈值化前）确认 ∈ [0,1]
□ grid_sample 前后 inf 占据区域是否对齐（核验 0.5m/0.512m 网格错位影响）
□ 保存典型样本 P_v, P_inf, P_f, Y 做可视化
```

---

**报告修正完成** ✓（7 处技术点已对照 agent-fusion-improved 分支源码更正）
