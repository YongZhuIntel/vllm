# 冷专家 iGPU 容量模式(capacity mode)

> 目的:**让装不下的 MoE 模型装得下**,不是提速。
> 与 `STATUS.md` 里那套 perf 卸载(`VLLM_XPU_IGPU_MOE=1`)是两个东西,互斥。
>
> 落点:`vllm/model_executor/layers/fused_moe/igpu_moe_capacity.py`(新)+
> `layer.py` / `ipex_quant.py` / `unquantized_fused_moe_method.py` 三处钩子。

---

## 1. 它做什么

把每个 MoE 层的 256 个专家按 index 切成两半:

```
[0, E-K)   hot   → dGPU(B60 显存)
[E-K, E)   cold  → iGPU sidecar 进程(= 系统内存)
```

关键在于**独显从头到尾不会 materialize 冷专家**:`FusedMoE.__init__` 把
`local_num_experts` 改成 `E-K`、`_expert_map` 里冷专家置 -1,于是所有 quant method 的
`create_weights` 只分配 `E-K` 份;`weight_loader` 里原本因 -1 被丢弃的分片改为写进
sidecar 的暂存区。没有"先全量加载再裁剪"的中间峰值。

正确性靠 ipex 的 `experts_start_id` 语义:两边都用**全局** logits 做 top_k 和 renorm,
各自只算自己持有的专家,相加 == 完整 MoE。

## 2. 为什么比 PP=2 更适合这台机器

2026-07-27 的 PP 尝试(`kv-block-zeroer-igpu-pp-debug-20260727.md`)卡在系统内存:
核显要吃 21.6 GiB(18 层的权重 + KV + oneCCL),`MemAvailable` 只剩 0.28 GiB,一发请求就 OOM。

| | PP=2 按层切 | 按专家 index 切 |
|---|---|---|
| 核显承担 | 18 层的**全部**(attn + 专家 + KV + oneCCL)= 21.6 GiB | **只有冷专家权重**,零 KV、零 attention |
| 调节粒度 | 层(最细 ~0.9 GB) | 专家(最细 120 MiB) |
| 实际余量 | 0.28 GiB | ~8 GiB(见下) |

## 3. 内存账(Qwen3.6-35B-A3B,E=256 H=2048 I=512,40 层,fp8)

```
每专家       = w13[1024,2048] + w2[2048,512] = 3,145,728 元素
             = 3.0 MiB (fp8) / 6.0 MiB (fp16)
每个专家 index 跨 40 层 = 120 MiB (fp8)
全部 256 个   = 30.0 GiB (fp8)   ← 占整个模型的 92%
```

**K 怎么定**(K = 放到 iGPU 的专家数):

```
hot 预算(GiB) = 23.91×gpu_util − 非专家权重 − KV − workspace
K             = 256 − hot 预算 / 0.117 GiB
```

代入本机(gpu_util=0.95、非专家 ≈6.3 GiB[^1]、8k 上下文 KV=0.63 GiB、workspace ≈1.0 GiB):

```
hot 预算 = 22.7 − 6.3 − 0.63 − 1.0 = 14.8 GiB  →  hot ≈ 126,  K ≈ 130
```

| | 大小 |
|---|---:|
| dGPU:非专家 + 126 个热专家 + KV + workspace | ~22.7 / 23.91 GiB |
| **iGPU(系统内存):130 个冷专家** | **15.2 GiB** |
| 注册期暂存 shm(一层的 bf16,注册完自动释放) | 0.76 GiB(瞬时) |
| 两个 worker + sidecar 的 anon | ~5.5 GiB |
| 桌面等 | ~2.4 GiB |
| **系统内存合计** | **~23 / 30.96 GiB → 余 ~8 GiB** |

[^1]: 由 PP 那次实测反推:fp8 全模型 19.26+17.05 = 36.31 GiB,减去专家 30.0 GiB。

**先从 `COLD_K=130` 起步**:独显放不下是启动即失败的硬错误,而系统内存这边有 8 GiB 余量。
跑通之后再往下调 K 换速度(K 每减 10,独显多吃 1.17 GiB,decode 每层少 ~23 µs)。

## 4. 性能(实测,单个 MoE 层,K=115,`test_capacity_split.py --bench`)

| | 冷侧分支 | hot‖cold 每层墙钟 | dGPU 全量(不卸载) | 倍数 |
|---|---:|---:|---:|---:|
| DECODE M=1 | 359 µs | **406 µs** | 139 µs | 2.9× |
| PREFILL M=1024 | 36.4 ms | **37.8 ms** | 2.7 ms | 14× |

外推到 40 层:

- **decode ≈ +10.7 ms/token**(相对全放独显),量级上落在 ~60 tok/s。可用。
- **prefill ≈ 1.5 s / 1024 token**(~680 tok/s)。**慢,这是这套方案的主要代价。**
  原因是 iGPU 在 `I=512`、每个专家只摊到 ~32 个 token 的小 GEMM 上利用率极低
  (实测 ~0.7 TFLOP/s,远低于 STATUS §7.3 里 `I=768` 时的 ~17 TFLOP/s)。
  搬运只占 ~3 ms,不是瓶颈 —— 换 pinned 传输收益有限。

> 想要快,唯一的结构性出路是 **int4 专家全放独显**(~21 GiB,不需要核显),见
> 本仓外的讨论;容量模式解决的是"没有 int4 checkpoint 时也要能跑"。

## 5. 启动

```bash
# —— 系统准备(不做还是会 OOM)——
sync; echo 3 > /proc/sys/vm/drop_caches   # 加载留下的 page cache 会长期占 MemAvailable
systemctl stop gdm                        # 或杀 gnome-shell/gjs/ibus/code-server,腾 1.5~2 GB
# swap 建议加到 32 GiB:Shmem 不能像 page cache 那样丢弃,只能换出

VLLM_XPU_IGPU_MOE_CAPACITY=1 \
VLLM_XPU_IGPU_MOE_COLD_K=130 \
VLLM_XPU_IGPU_MOE_COLD_QUANT=fp8 \
VLLM_XPU_IGPU_MOE_MASK=1 \
VLLM_XPU_IGPU_MOE_DEBUG=1 \
VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTHONPATH=/llm/zhuyong/vllm \
python3 -m vllm.entrypoints.openai.api_server \
  --model /llm/zhuyong/Qwen3.6-35B-A3B \
  --quantization fp8 --dtype float16 --enforce-eager \
  --gpu-memory-util 0.95 --block-size 64 \
  --max-model-len 8192 --max_num_batched_tokens 1024 \
  --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0}' \
  --port 8000
```

### 硬性约束

| 约束 | 为什么 |
|---|---|
| **不能** 同时开 `VLLM_XPU_IGPU_MOE=1` | 那是 perf 模式,会在独显上**再放一份**冷专家,正好相反。已加启动期检查。 |
| `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0` | `=1` 那条 legacy 路径把权重建在 CPU 上,而 ipex 的 fp8 量化算子只有 XPU 实现 → `Could not run 'torch_ipex::dynamic_scaled_fp8_quant.xpu' ... from the 'CPU' backend`。**这是 fork 里已有的问题,与容量模式无关**,但你的 shell 里当前就是 `=1`,注意覆盖。 |
| `DISABLE_ESIMD_MOE=1` | 代码会**自动强制**并打 WARNING。`Qwen3NextSparseMoeBlock.forward` 在 fp8 且 `num_tokens<=128`(即全部 decode)时走 ESIMD 融合算子,直接读 `w13_weight` 且按**全局** 256 索引 —— 容量模式下独显只有 E-K 份,那条路会越界或静默只算半个 MoE。代价是 decode 失去 ESIMD 快路径。 |
| `tp=dp=ep=pp=1` | 冷切分本身就是一个"不均匀的 EP rank",和真 EP 叠加没实现,启动期直接报错。 |
| `--enforce-eager` | 冷侧是跨进程同步调用,没在 torch.compile 图里验证过。 |
| 只支持**非预量化** checkpoint | 冷侧只会吃裸 bf16/fp16 专家权重再自己量化。遇到 `*_scale` / `g_idx` 会明确报错,不会算错。 |
| sidecar 挂了 = 硬失败 | 冷专家只存在于那个进程,**无法回退**。代码会抛清晰的异常而不是静默算半个 MoE。 |

### KVBlockZeroer 补丁

`kvblockzeroer-index-fix.patch`(KV-cache-group 下标 vs layer 下标混用)**已经在
`/llm/zhuyong/vllm` 这棵树里了** —— `gpu_model_runner.py` 有 `_kv_caches_by_layer()` /
`_mamba_state_tensors`,`utils.py` 的 `KVBlockZeroer` 已改成按 layer name 索引的 dict。
不需要再打。

注意 `/llm/vllm` 是**另一棵独立的树**(不是软链),没有本次改动;跑的时候
`PYTHONPATH` 必须指向 `/llm/zhuyong/vllm`。系统里还装了一份 pip 的
`/usr/local/lib/python3.12/dist-packages/vllm`(0.14.1.dev0),PYTHONPATH 能盖住它。

## 6. 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `VLLM_XPU_IGPU_MOE_CAPACITY` | `0` | 总开关 |
| `VLLM_XPU_IGPU_MOE_COLD_K` | `-1` | 绝对冷专家数;`<0` 时用 `COLD_FRAC` |
| `VLLM_XPU_IGPU_MOE_COLD_FRAC` | `0.45` | 冷专家比例 |
| `VLLM_XPU_IGPU_MOE_COLD_QUANT` | `fp8` | 冷权重在 iGPU 上的 dtype(`fp8` / `none`) |
| `VLLM_XPU_IGPU_MOE_MASK` | `1` | sidecar 的 `ZE_AFFINITY_MASK` |
| `VLLM_XPU_IGPU_MOE_REG_CHUNK` | `8` | 注册时 iGPU 一次吞几个专家(控制瞬时显存) |
| `VLLM_XPU_IGPU_MOE_MAX_INFLIGHT` | `4` | 允许同时在加载的 MoE 层数(每层一份 CPU 暂存 ≈818 MiB @ K=130)。本模型排序后真实峰值是 3,见 §10.1(c) |
| `VLLM_XPU_IGPU_MOE_COLD_PERM` | `0` | **别开**,见下 |
| `VLLM_XPU_IGPU_MOE_DEBUG` | `0` | 日志 |

### 关于 `COLD_PERM`

`igpu_moe_offload.py`(perf 模式)用的是"把冷专家的 logits 列置换到最前 +
`experts_start_id=0`"。**实测这个做法有 bug**:置换改变了 top_k 的并列打破顺序和 softmax
的求和次序,在 `E=256 / top_k=8 / fp16 logits` 下,**每 1024 个 token 有 ~6 个**会被
hot / cold 两半选中不同的专家,那几行结果整行错掉(`max|diff|` 0.31,rel 0.15)。

容量模式默认改用 `experts_start_id = E-K` **不置换**,同样条件下 `bad_rows = 0/1024`、
rel = 4.8e-04(纯跨设备数值噪声)。

> perf 模式那边同样受影响。STATUS §9.1 记的"max diff 0.002"是在 E=64 下测的,
> 概率低所以被当成了数值噪声 —— 如果还要用 perf 模式,建议一并改掉。

## 7. 自测

两个脚本都不需要 35B 权重,几分钟跑完,真机真跨进程:

```bash
cd /llm/zhuyong/vllm

# a) sidecar 本身:共享内存协议 / 掩码 / 分块 fp8 注册 / 前向
PYTHONPATH=$PWD python3 igpu_moe_offload/test_capacity_split.py
PYTHONPATH=$PWD python3 igpu_moe_offload/test_capacity_split.py \
    --experts 256 --cold-k 114 --hidden 2048 --inter 512 --top-k 8 \
    --prefill-tokens 1024 --quant fp8 --bench      # 带性能数字

# b) vLLM 集成路径:expert_map 切分 / weight_loader 分流 / apply 钩子
PYTHONPATH=$PWD python3 igpu_moe_offload/test_capacity_layer.py
PYTHONPATH=$PWD python3 igpu_moe_offload/test_capacity_layer.py --fp8
PYTHONPATH=$PWD python3 igpu_moe_offload/test_capacity_layer.py --fp8 \
    --experts 256 --hidden 2048 --inter 512 --top-k 8 --tokens 512
```

已跑过的结果(2026-07-28,B60 + Arrow Lake iGPU):全部 `RESULT: PASS`,
`bad_rows=0`,sidecar 日志确认落在 `Intel(R) Graphics`(核显)而不是独显。

测试里有两道防假阳性的闸:
- 冷侧输出必须非零、且 `hot` 单独算出来必须 ≠ 参照(否则"通过"没有意义);
- 区分 `bad_rows`(选错专家,整行错)与 `rel`(数值噪声)—— 这两个是完全不同的病。

## 8. Runbook —— 下次直接照抄

### 8.1 先跑自测(5 分钟,不碰 35B,确认环境没变)

```bash
cd /llm/zhuyong/vllm
PYTHONPATH=$PWD python3 igpu_moe_offload/test_capacity_split.py            # 期望 RESULT: PASS
PYTHONPATH=$PWD python3 igpu_moe_offload/test_capacity_layer.py --fp8      # 期望 RESULT: PASS
```

对照基线:`/llm/zhuyong/igpu_moe_offload/capacity-mode-20260728/test-results-20260728.txt`
(逐字记录了 2026-07-28 的全部输出与性能数字)。

必须看到 `[igpu-cold] up on 'Intel(R) Graphics'` —— 如果是 `Arc(TM) Pro B60`,说明
`ZE_AFFINITY_MASK` 没生效,冷专家会去吃独显显存,整套方案失效(代码会额外打 WARNING)。

### 8.2 起服务前的准备

```bash
pkill -f "api_server|VLLM::" 2>/dev/null
sync; echo 3 > /proc/sys/vm/drop_caches      # 加载权重留下的 page cache 会长期占着 MemAvailable
grep -E "^(MemAvailable|Shmem):" /proc/meminfo   # 期望 MemAvailable ≳ 28 GiB、Shmem ~100 MB

# swap 只有 8 GiB 且常被占掉一大半;首请求的 Triton JIT 有几百 MB 峰值,建议加一块
fallocate -l 24G /swap2.img && chmod 600 /swap2.img && mkswap /swap2.img && swapon /swap2.img
```

### 8.3 起服务(这就是那条命令)

```bash
cd /llm/zhuyong/vllm

VLLM_XPU_IGPU_MOE_CAPACITY=1 \
VLLM_XPU_IGPU_MOE_COLD_K=130 \
VLLM_XPU_IGPU_MOE_COLD_QUANT=fp8 \
VLLM_XPU_IGPU_MOE_MASK=1 \
VLLM_XPU_IGPU_MOE_DEBUG=1 \
VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTHONPATH=/llm/zhuyong/vllm \
python3 -m vllm.entrypoints.openai.api_server \
  --model /llm/zhuyong/Qwen3.6-35B-A3B \
  --quantization fp8 --dtype float16 --enforce-eager \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 --max-num-batched-tokens 1024 --block-size 64 \
  --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0}' \
  --port 8000
```

**不要带**:`-pp=2`、`VLLM_XPU_IGPU_PP`、`VLLM_PP_LAYER_PARTITION`、`CCL_PLUGIN`、
`TORCH_LLM_ALLREDUCE`(容量模式是单 rank,带 PP 会在 `_init_igpu_cold_split` 直接报错);
`VLLM_XPU_IGPU_MOE=1`(perf 模式,互斥,已有检查拦);
`--enable-prefix-caching`(首跑先别开,mamba state checkpoint 会把 KV 撑到约 4 倍)。

`PYTHONPATH` 必须是 `/llm/zhuyong/vllm` —— `/llm/vllm` 是另一棵独立的树,没有这些改动。
`VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0` 必须显式给 —— 这台机器的 shell 里预置了 `=1`。

### 8.4 加载期盯这几行

```
VLLM_XPU_IGPU_MOE_CAPACITY: 126/256 experts per layer stay on the dGPU, 130 are offloaded
WARNING ... forcing DISABLE_ESIMD_MOE=1                       ← 正常,预期内
[igpu-cold] up on 'Intel(R) Graphics' (ZE_AFFINITY_MASK=1)    ← 必须是核显
[igpu-cold] layer N registered                                 ← 应出现 40 次
[igpu-cold] registration done: 40 layers, xpu alloc=15.xx GiB  ← 层数必须是 40
```

另开一个终端:`watch -n2 'grep -E "^(MemAvailable|Shmem):" /proc/meminfo; xpu-smi stats -d 0 | grep "Memory Used"'`

### 8.5 冒烟

```bash
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"/llm/zhuyong/Qwen3.6-35B-A3B","prompt":"The capital city of France is",
       "max_tokens":32,"temperature":0}'
```
首个请求会慢(Triton JIT + prefill 本身就慢),不是卡死。

### 8.6 出错怎么调

| 症状 | 处理 |
|---|---|
| 加载时独显 OOM / `No available memory for the cache blocks` | `COLD_K` +10(独显每多卸 10 个专家省 1.17 GiB) |
| `MemAvailable` 掉到 1 GiB 以下、Worker 被 OOM killer 杀 | `COLD_K` −10,并关掉 vscode-server / 桌面 |
| `registration done: N layers` 且 N < 40 | 有 MoE 层的冷权重没到齐,查日志里 `layer X registered` 缺哪个 |
| `sidecar died` / `reported an error` | 看 sidecar 自己的 traceback(直接打到同一个 stdout)。容量模式**不能回退**,必然是硬失败 |
| `up on 'Intel(R) Arc(TM) Pro B60'` | 掩码没生效,检查 `VLLM_XPU_IGPU_MOE_MASK` 与 `sycl-ls` 的设备序 |
| `Could not run 'torch_ipex::dynamic_scaled_fp8_quant.xpu' ... 'CPU' backend` | 忘了 `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0` |
| decode 结果明显不对但不报错 | 检查 `DISABLE_ESIMD_MOE` 是否真的是 1;再跑一遍 §8.1 的自测看 `bad_rows` |

### 8.7 代码快照

`/llm/zhuyong/igpu_moe_offload/capacity-mode-20260728/capacity-mode.patch`
(基于 `403f4e6b1 "add moe split"`,含 4 个新文件 + 3 处改动)

```bash
cd /llm/zhuyong/vllm
git apply --check /llm/zhuyong/igpu_moe_offload/capacity-mode-20260728/capacity-mode.patch
git apply         /llm/zhuyong/igpu_moe_offload/capacity-mode-20260728/capacity-mode.patch
```

## 9. 已知未做

1. ~~**端到端没跑过 35B**~~ —— 2026-07-29 跑通了,见 §10。
2. **prefill 慢 14×**。想改善只能减小 K,或者换 int4 方案绕开核显。
3. 传输仍是 pageable numpy。按 STATUS §9.5,pinned 能把 4 MiB 的 D2H 从 ~1 ms 压到
   ~0.16 ms —— 但相对 36 ms 的 iGPU 计算不值一提,优先级低。
4. 预量化(AWQ/GPTQ)checkpoint 的冷侧摄入没实现,会明确报错。
5. `XPUGPTQMarlinMoEMethod` 没挂钩子(int4 走独显单卡就够,不需要容量模式)。

---

## 10. 端到端实测(2026-07-29,Qwen3.6-35B-A3B,K=130)

**跑通了。** §8.3 那条命令原封不动可用,但要先带上下面三个修复 —— 三个都是
合成单层测试**结构上测不到**的东西,前两个直接让服务起不来。

### 10.1 三个修复

**(a) `ipex_quant.py` —— fp8 流式量化的记账被冷专家撞崩**

`XPUFp8MoEMethod.create_weights` 里的 `patched_weight_loader` 对**每一个**分片记账。
热的那半(126 个专家)一装满就立刻量化并 `del layer._w13_loaded_numel`,
而此时冷专家的分片还在陆续到达 —— 下一个冷分片进到
`layer._w13_loaded_numel += ...` 就是:

```
AttributeError: 'SharedFusedMoE' object has no attribute '_w13_loaded_numel'
```

修法:`patched_weight_loader` 开头识别出映射到 -1 的冷分片,直接透传给
`orig_weight_loader`,不参与流式量化的任何记账。

**(b) `test_capacity_layer.py` —— 自测有假阴性,(a) 就是这么漏掉的**

原来的 harness 每个分片都重新 `getattr(param, "weight_loader")`。但 fp8 流式路径
materialize 的时候用**原始** loader 重新 `register_parameter` 了 `w13_weight`
(`ipex_quant.py` 里 `orig_attrs["weight_loader"] = orig_weight_loader`),
于是**第二个分片起就绕过了 `patched_weight_loader`**,整条流式量化 + 冷热交错的
路径一次都没测到。真实的 `Qwen3_5Model.load_fused_expert_weights` 是
`param = params_dict[name]` + `param.weight_loader` 只取一次、对 E 个专家复用,
所以只有真机会崩。

修法:harness 改成 param 和 loader 都只解析一次。改完之后**不打 (a) 的补丁就能
复现真机那个 AttributeError**,这个自测才真的是一道闸。

**(c) `default_loader.py` —— `glob.glob` 不排序,把 in-flight 撑爆**

`_prepare_weights` 里 `hf_weights_files += glob.glob(...)` 是目录顺序,**任意**。
本模型每层的 `gate_up_proj` 和 `down_proj` 常常在不同 shard 里,一层要等两个 shard
都到齐才能注册;shard 顺序一乱,这个窗口就能横跨整个 checkpoint:

```
实测(未排序):层的到达顺序 10, 8, 9, 26, 27, 38, 39, 14, 15, 16, 2, 3, 23, 24, 25, 0 ...
             → in flight 一路涨到 6+,MAX_INFLIGHT 直接报错
RuntimeError: igpu cold: more than 2 MoE layers are loading concurrently
              (in flight: [26, 27], new: 38)
```

修法:`hf_weights_files.sort()`。零填充的文件名排序即数字序,窗口降到最小。
(多线程加载默认关闭,所以这不是竞态,纯粹是文件顺序。)

**顺带把 `MAX_INFLIGHT` 默认值 2 → 4**:排好序之后本模型**真的**会同时开 3 层
(file 8 里放着 layer 11/12/13 的 gate_up,它们的 down_proj 在 file 9),
默认 2 在健康 checkpoint 上也会失败。每层在飞 = `cold_k*(w13+w2)` 的 param dtype
暂存 ≈ **818 MiB**(K=130),这是个真的主存旋钮。

### 10.2 实测数字

| | 预测(合成单层外推) | **实测(35B 真机)** |
|---|---:|---:|
| iGPU 冷专家 alloc | 15.2 GiB | **15.23 GiB** ✓ |
| dGPU 占用 | ~22.7 GiB | **23.9 GiB**(24451 MiB) |
| KV cache | — | **22,528 tokens** |
| 稳态 `MemAvailable` | ~8 GiB | **6.0 GiB**(另有 6 GiB 换出到 swap) |
| decode | ~60 tok/s(+10.7 ms/tok) | **45.4 tok/s(22.0 ms/tok)** |
| prefill | ~680 tok/s | **403 tok/s**(1270 tok / 3.15 s) |

加载耗时约 25 分钟(67 GB bf16 checkpoint,边读边在线量化)。

启动日志确认(§8.4 那几行全部对上):

```
VLLM_XPU_IGPU_MOE_CAPACITY: 126/256 experts per layer stay on the dGPU, 130 are offloaded
WARNING ... forcing DISABLE_ESIMD_MOE=1
[igpu-cold] up on 'Intel(R) Graphics' (ZE_AFFINITY_MASK=1) E=256 K=130 H=2048 I=512 quant=fp8
[igpu-cold] registration done: 40 layers, xpu alloc=15.23 GiB
```

冒烟输出连贯("The capital city of France is" → " Paris, a city that has been the
political, economic, and cultural center of the country for centuries. ..."),
128 token 长生成也连贯,服务 `/health` 200,启动后日志零 ERROR。

**结论:decode 45 tok/s 可用,prefill 403 tok/s 是主要代价 —— 比合成外推还要再慢
1.7×。§4 的结论不变:想要快,出路是 int4 全放独显,不是调这套。**

### 10.3 复现

日志:`/llm/zhuyong/logs/e2e-20260729.log`(成功),
`e2e-20260729-run{1,2,3}-*.log`(三次失败,分别对应 (a)、(c)、(c) 放大 MAX_INFLIGHT 后
观察到的乱序)。启动脚本:`/llm/zhuyong/logs/serve-capacity.sh`。
