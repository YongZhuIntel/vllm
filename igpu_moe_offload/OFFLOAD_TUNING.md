# 冷专家 iGPU 卸载数调优指南(跨平台)

> 目的:给定一台「dGPU + iGPU 同主机」的机器,确定 **prefill 阶段** 每层把多少个冷专家
> 卸载到 iGPU(记为 **K***)能让每层耗时最短、prefill 最快。方法与工具平台无关,
> 换机器只需重跑扫描。
>
> 前置结论(见 STATUS.md):**只在 prefill 卸载**(decode 是内存带宽 bound,卸载 no-go)。
> 本文只讲 prefill 的 K* 怎么找。

---

## 1. 核心模型:为什么存在最优 K*(U 形曲线)

每个 MoE 层在 prefill 时:

- **dGPU 干**:attention(**不可卸载**)+ 热专家 + `(cold_active − K)` 个冷专家。
- **iGPU 干**(与 dGPU 并行):搬运 + `K` 个冷专家。
- 每层墙钟 = `overlap_RT(K) = max( dGPU_own(K),  transport + iGPU_compute(K) )`
- **baseline = overlap_RT(0)**(dGPU 全干,iGPU 空闲)。

两条线随 K 变化:

```
dGPU_own(K)   = A + (hot + cold_active − K)·d      # 随 K 线性下降(卸走一个省 d)
iGPU_side(K)  = transport + K·g                    # 随 K 线性上升(多算一个加 g)
```

- `A`         = dGPU attention 时间(不可卸载,固定阴影)
- `d`         = **dGPU** 每个专家 FFN 耗时
- `g`         = **iGPU** 每个专家 FFN 耗时(通常 `g > d`)
- `transport` = 一次 shm 往返搬运耗时(与 K 无关,只看每专家行数 M)

**K < K***:dGPU 是关键路径,K↑ → RT↓(卸载有用)。
**K > K***:iGPU 变成关键路径,K↑ → RT↑(iGPU 太慢拖后腿)。
**K***:两条线相交处,`dGPU_own(K*) ≈ iGPU_side(K*)`,RT 最小 → **U 形谷底**。

---

## 2. 决定 K* 的四个平台基本量

| 量 | 含义 | 怎么读(用扫描工具的列) |
|---|---|---|
| `A` | dGPU attention(不可卸载) | `baseline_dGPU_own − (hot+cold_active)·d` |
| `d` | dGPU 每专家 FFN | `dGPU_own` 列随 K 的斜率 Δ/专家 |
| `g` | iGPU 每专家 FFN | `iGPU_cmp` 列随 K 的斜率(或 `iGPU_cmp/K`) |
| `transport` | shm 往返 | `iGPU_side − iGPU_cmp`(任一 K>0 行,应大致恒定) |

**最关键的是比值 `g/d`**:iGPU 相对 dGPU 越慢,能藏进阴影的专家越少,K* 越小。
`A`(attention 阴影)越大 → 能藏更多 → K* 越大。

---

## 3. 找 K* 的两种方法

### 方法 A:直接扫描(推荐,最可信)

跑 `moe_offload_prototype.py` 的 `--sweep-offload`,读 `overlap_RT` 最小那行即 K*。
工具会直接打印 `[RESULT] optimal K*=... speedup=...`。先粗扫再在谷底附近细扫。

### 方法 B:解析预测(理解 / 快速估算,不用全扫)

令两条线相交:
```
A + (hot+cold_active−K)·d  =  transport + K·g
⟹  K* ≈ [ dGPU_own(0) − transport ] / ( d + g )
```
其中 `dGPU_own(0) = A + (hot+cold_active)·d` = baseline 的 dGPU 纯计算(K=0 行的 dGPU_own)。

> 本机实测校验:`dGPU_own(0)=7421µs, transport≈1000µs, d≈66µs, g≈297µs`
> ⟹ `K* ≈ (7421−1000)/(66+297) = 17.7`,实测扫描 K*=16 —— 吻合。

**加速上限估算**:`speedup ≈ baseline_RT / overlap_RT(K*)`,其中
`overlap_RT(K*) ≈ dGPU_own(0) − d·K* + 少量同步开销`。
直觉:iGPU 慢 `r = g/d` 倍时,它最多吃下约 `1/(r+1)` 的**专家**工作,
所以加速天花板 ≈ (专家工作占 dGPU 总工作的比例) × 该分数。本机 ~20%。

---

## 4. 新平台操作步骤

1. **确认双进程可用**:两块 XPU 能各自 `ZE_AFFINITY_MASK=0/1` 钉住
   (import torch 前设)。`sycl-ls` 应看到 dGPU 与 iGPU 两块 level_zero 设备。
2. **填模型参数**:`--hidden H`、`--inter I`、`--top-k`、总专家数、每层热专家数。
   - `--cold-active` = 每层 active 的冷专家数(≈ 总专家 − 热专家)。
   - `--cold-experts` ≥ `--cold-active`(iGPU 常驻池,可设成总专家数)。
   - 每专家行数由工具自动算:`M = T·top_k / (hot+cold_active)`(**不要**手改成 M=T)。
3. **粗扫**:`--sweep-offload "0,4,8,16,24,32,48"`(0 必须在,作 baseline)。
4. **读 U 形谷底**:找 `overlap_RT` 最小的 K;若最小值在边界,向外扩 K 再扫。
5. **细扫**:在谷底 ±4 内细扫定 K*。
6. **记录**:K*、卸载比 `K*/cold_active`、`speedup`。
7. **灵敏度**(可选):`--attn-scale 2` 看 attention 更重时 K* 右移多少,给收益包络。

---

## 5. 命令与输出解读

```bash
cd <this_dir>
# 主扫描(batched 用 bmm 减少 launch;非 batched 走逐专家循环)
python moe_offload_prototype.py --mode compute --batched \
    --hidden 2048 --inter 768 --top-k 8 \
    --hot-experts 8 --cold-active 56 --cold-experts 64 \
    --sweep-offload "0,8,12,16,20,24,28" --prefill-tokens 2048
```

输出列:
```
 K   dGPU_own   iGPU_cmp  iGPU_side  transport  overlap_RT
 0    7421µs      0         0          0          8114µs   <- baseline
16    6366µs    5055µs    5933µs      878µs       6780µs   <- 谷底 = K*
```
- `dGPU_own`:dGPU 自身工作纯设备时间(xpu.Event 计),随 K 降。
- `iGPU_cmp`:iGPU 纯计算(Event 计),随 K 升;斜率 = `g`。
- `iGPU_side`:iGPU 全程 wall(含搬运),= `transport + iGPU_cmp`。
- `transport = iGPU_side − iGPU_cmp`,应大致恒定(只看 M)。
- `overlap_RT`:**每层墙钟**(= max 两边),这是要最小化的量。
- 谷底那行的 K 就是 K*;`[RESULT]` 行给 K* 与 speedup。

---

## 6. 测量正确性(移植时务必保留,否则数字失真)

这两点是本原型踩过的坑,换平台/改代码要保住:

1. **真并行时序**:先发请求 + 提交固定量 dGPU 工作(async),**再**自旋等 iGPU,
   **最后只 synchronize 一次**。RT = `max(dGPU own, iGPU 往返)`。
   ❌ 绝不能用「按时间 deadline 的 busy-loop 提交 kernel + synchronize」——
   会无界堆积 async 队列、尾部 drain,RT 虚高、数字无意义。
2. **xpu.Event 单位**:`elapsed_time()` 返回**毫秒**,转 ns 要 `×1e6`(别写成 ×1e3)。
   校验:event 时间应与同段 wall 时间接近(如 6176µs vs 6727µs)。

---

## 7. 何时卸载不值得(先判断再做)

- **dGPU 每专家很便宜 + iGPU 极慢(g/d 很大)**:U 形谷底几乎贴着 baseline,加速 <5%,
  不值得(例:cold_active 很少 → M 很大 → dGPU 高利用、iGPU 反而低效,gap 拉大)。
- **attention 阴影 A 很小**(短序列 / 小模型):没多少阴影可藏冷专家,K* 小、收益低。
- **decode**:恒为带宽 bound,任何 K 都 no-go(见 STATUS §7)。

经验:`g/d` 越小(iGPU 相对不那么慢)、`A` 越大(attention 越重)、专家工作占比越高,
卸载越划算。本机在**现实 prefill(多专家、小 M)**下 `g/d≈4.5`,收益 ~20%。

---

## 8. 已知近似 / 真机复核

原型用了几个简化,**真机集成后必须复核**:
- **uniform token 分摊**:每专家都按平均 `M` 行,忽略路由**负载不均**(尾专家 2–3×),
  会**略高估**加速。真实需按实际 routing 直方图评估尾专家。
- **单层孤立测量**:未含连续多层 pipeline、逐层同步/依赖的开销。
- **搬运用同一 M 行 expand 给 K 个专家**:低估真实 gather/scatter 的搬运量
  (但搬运本就占比小)。
- 结论方向(prefill go / decode no-go)稳健;**具体百分比需在真模型端到端复核**。
