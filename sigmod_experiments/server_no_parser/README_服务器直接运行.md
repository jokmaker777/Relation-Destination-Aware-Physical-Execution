# SSH 服务器直接运行说明（无命令行参数）

这些脚本不使用 argparse。图路径、GPU 编号和输出目录全部位于
`settings_no_parser.py`。默认工业图为：

```text
/path/to/artifact_workspace/cache/graph_20241129.pkl
```

进入目录：

```bash
cd /path/to/artifact/sigmod_experiments/server_no_parser
```

随后每次只需运行一条 `python 文件名.py`。脚本会跳过已经完成的结果，SSH
中断后可以重新执行同一个脚本继续。

## 顺序

### 0. 图和服务器预检查

```bash
python 00_preflight_no_parser.py
```

读取工业图、自动统计真实 labeled-edge 数、检查原训练脚本，并确认进程能
看到四张 GPU。后续脚本直接读取这里生成的 metadata，不需要手工填边数。

### 1. 1/2/3/4 卡 exactness

```bash
python 01_correctness_1to4gpu_no_parser.py
```

比较分布式结果与完整边集参考。四种卡数全部打印 `exactness PASSED` 后才能
继续。

### 2. 工业训练 pilot

```bash
python 02_industrial_pilot_no_parser.py
```

在 20241129 上运行 1/2 卡、blocking/async 的四个小规模端到端任务，检查
数据加载、DDP、loss、显存和计时文件。

### 3. 工业 1/2/3/4 卡主实验

```bash
python 03_industrial_scale_1to4gpu_no_parser.py
```

共 24 个组合：四种卡数、三种布局、两种通信计划。第 2 步完成的四个任务
会自动跳过。

### 4. 4 卡 B/C 联合搜索

```bash
python 04_joint_tuning_4gpu_no_parser.py
```

共 40 个组合：两种 superblock assignment、四个 B、五种 C/通信计划。

### 5. 首次汇总和计划排名

```bash
python 06_collect_and_rank_no_parser.py
```

生成工业汇总和 `single_graph_4gpu_plan_ranking.csv`。由于目前只有一个工业
图，这只是单图 plan ranking，不能称为 leave-one-graph-out cost model。

### 6. 给最佳计划和主基线补三次重复

```bash
python 07_repeat_top_plans_no_parser.py
```

自动读取排名，对前三名计划和直接边均衡 blocking 基线补 repeat 2/3。
完成后再次运行：

```bash
python 06_collect_and_rank_no_parser.py
```

### 7. 公开数据与合成图

```bash
python 05_public_operator_matrix_no_parser.py
```

依次运行 HGB-DBLP、uniform synthetic、skewed synthetic、ogbn-mag；覆盖
五种布局、B/C、blocking/async 和 1/2/3/4 卡。该阶段较长，随时停止后重新
执行会从未完成任务继续。

公开实验完成后最后再运行一次：

```bash
python 06_collect_and_rank_no_parser.py
```

## 12 个工业日期的压缩实验（当前优先）

已经完成 `00`/`01`/`02` 后，暂时不必先跑单日期的 `03`/`04`。
先用多日期实验补足跨快照稳健性证据。

今晚可只运行一条：

```bash
python start_multidate_overnight_no_parser.py
```

该入口先执行 `08_multidate_preflight_no_parser.py`，检查和统计
20241124--20241129 及 20241213--20241218 共 12 个图；通过后自动
执行 `09_overnight_multidate_ablation_no_parser.py`。

`09` 共 72 个四卡任务：12 个日期 x 3 种布局 x
blocking/async 两种通信计划。完成结果会自动跳过，中断后重新
运行同一命令即可续跑。单个配置失败会记录并继续；连续 3 个
失败才会停止，避免系统性错误反复空跑。

查看已完成数量：

```bash
python check_multidate_status_no_parser.py
```

多日期首轮完成后：

```bash
python 10_collect_multidate_no_parser.py
python 11_repeat_multidate_top_plans_no_parser.py
python 10_collect_multidate_no_parser.py
```

`10` 按日期等权汇总并生成四卡计划排名；`11` 对前三名和基线在
4 个代表日期上补 repeat 2/3，之后再跑 `10` 合并结果。

## 输出位置

所有结果位于：

```text
/path/to/artifact_workspace/sigmod2027_20241129
```

主要文件：

- `industrial_measurements.csv`：工业端到端汇总；
- `operator_measurements.csv`：公开/合成图算子汇总；
- `single_graph_4gpu_plan_ranking.csv`：50-epoch horizon 下的四卡计划排名；
- `multidate_graph_metadata.json`：12 个工业图的规模与关系统计；
- `multidate_measurements.csv`：多日期端到端原始汇总；
- `multidate_4gpu_plan_ranking.csv`：按日期等权的四卡计划排名；
- `launcher_logs/`：每个任务的完整日志。

## 当前证据边界

一个 20241129 工业图可以完成正确性、布局消融、B/C 搜索和 1/2/3/4 卡
扩展性，但不能验证跨快照代价模型。最终若要保留“跨图计划选择”论点，仍需
加入至少两个其他工业快照，或把公开数据适配为相同的端到端计划选择实验。

