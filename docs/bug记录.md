# Bug 记录

修复过的 bug 的完整档案。格式：现象 → 根因 → 为何逃过测试 → 修复 → 防回归。
按发现顺序排列，每条标注修复提交。

---

## BUG-001 split→reduce 同 pass RAW race（隐蔽度最高）

**修复提交**：c456a6e（2026-10-06）

**现象**：无任何外部症状 —— 自检 40 项全 PASS，问答正确，端到端数字正常。
是在为「扩大 split-K 并行度」审查 `gemm()` 调度时人工发现的。

**根因**：`QwenGpt.gemm()` 的 split-K 分支中，`pGemvSplit` 与 `pGemvReduce`
两个 dispatch 之间没有 `passBreak()`。reduce 读的 `partial` 正是 split 写的
—— 同一 compute pass 内的 dispatch 之间**没有隐式内存屏障**，对同一 buffer
的 RAW（写后读）在 WebGPU 规范上是未定义行为。

**为何逃过测试**：未定义 ≠ 必然出错。Dawn/Vulkan 在该实现上恰好按提交顺序
串行执行了同 pass 的 dispatch，数值碰巧全对。这类 bug 的危险在于：
换驱动/换 GPU/驱动更新后可能随机出现数值错误，且极难复现定位。

**修复**：split dispatch 后显式 `passBreak()`；reduce 后也断开，使
「每个 split gemm 恒占 2 个 pass」（代价：+3 pass/层 ≈ +4% barrier 开销，
换来剖析布局可静态重建）。

**防回归**：布局一致性检查（BUG-003 引入）+ `runForward` 内注释明确
「同一 pass 内 dispatch 间不得存在对同一 buffer 的 RAW/WAR/WAW」。
审阅 GPU 调度代码时，把「同 pass 内的读写依赖」列为必查项。

---

## BUG-002 剖析：encoder finish 后再编码 resolveQuerySet

**修复提交**：d11390a（2026-10-06，同提交内修复）

**现象**：剖析测试首次运行 FAIL：
`[CommandEncoder] is already finished. - While encoding .ResolveQuerySet(...)`，
读回的时间戳全为 0。

**根因**：`CommandBatch.submit()` 先执行 `queue.submit([encoder.finish()])`
再调用 `encoder.resolveQuerySet(...)` —— encoder 在 finish 后即封闭，
不能再编码任何命令。

**修复**：调整顺序为 endPass → resolveQuerySet → copyBufferToBuffer →
finish → submit。resolve/copy 排在队列上 kernel 之后，
`readBatchProfile` 的 mapAsync 会等它们完成。

**防回归**：WebGPU 的 encoder 是一次性构建对象 —— 「所有编码必须发生在
finish 之前」应作为常识清单项；validation error scope（selftest 已有）
保证这类错误必然以 FAIL 形式暴露，不会静默。

---

## BUG-003 剖析布局漂移：labels=531 vs 实际 pass=459

**修复提交**：c456a6e（2026-10-06）

**现象**：扩展 split-K 档位后，剖析测试报
`剖析布局漂移：labels=531 vs 实际 pass=459`。

**根因**：剖析测试按静态假设重建 pass 布局（每层 12 pass），但实际调度中
`gemm()` 的 reduce dispatch 后没有 passBreak，导致 reduce 与**下一个 gemm
的 split 合并进同一个 pass**（该合并本身无数据冲突，合法但使 pass 数依赖
合并情况，无法静态预测）。

**为何逃过测试**：这是新加的防漂移检查**第一次运行就抓住的问题** ——
它防的正是这类「调度改了、剖析假设没跟上」的静默错位。

**修复**：① reduce 后补 passBreak，使每层 pass 数确定（每层 19 → 22，
总 531）；② 剖析测试改为按 `planGemvSplit`（已 export）动态重建布局，
并在布局不符时显式抛错而不是静默错位归类。

**防回归**：布局漂移检查长期保留。任何改变 pass 数量/顺序的调度改动
都会在剖析测试中显式失败。

---

## BUG-004 剖析等效带宽换算公式错误（显示 0.0 GB/s）

**修复提交**：d11390a（2026-10-06，同提交内修复）

**现象**：剖析 detail 输出「等效带宽 0.0 GB/s」—— 量纲错误
（`942 / gemvMs * 1.024 / 1e3` 少了 MB→GB 与 ms→s 的完整换算），
数字被缩小 1000 倍。

**根因**：手写单位换算无中间量校验。942 MiB ≈ 987.8 MB，
GB/s = 987.8 / (ms/1000) / 1000。

**修复**：改为两步清晰换算（weightMB → gemvBW），并保留
「峰值实测 20.2 / 带宽下限 48ms」作为对照锚点，读者可自行验证数量级。

**防回归**：带物理单位的换算写成命名中间变量 + 注释量纲；
输出里同时给出参照值（峰值/下限）便于人眼发现数量级异常。

---

## BUG-005 CommandBatch 初版编译错误（编译期拦截，未入库）

**发现提交**：974dd14 开发过程中（2026-10-06）

**现象**：tsc 报 `beginComputePass 不存在于 GPUDevice`、`this.pass 可能 null`。

**根因**：① `beginComputePass` 是 `GPUCommandEncoder` 的方法，不是
`GPUDevice` 的；② TS 对 `this.pass` 属性的 narrowing 在方法调用后失效。

**修复**：改用 `this.encoder.beginComputePass()`；dispatch 内用局部变量
承接 pass 引用再赋回字段。

**教训**：strict + noUnusedLocals 的 tsc 在这里把问题拦在了运行前 ——
保持 `tsc --noEmit` 作为提交前硬门槛的价值再次得到验证。

---

## BUG-006 .workbuddy 会话数据被误提交

**修复提交**：1cbee4d（2026-10-06）

**现象**：`git add -A` 后 `.workbuddy/`（AI 会话工作记忆目录）被带入提交。

**根因**：该目录在首次提交后才出现，`.gitignore` 未覆盖它；
`add -A` 对全部未跟踪文件生效。

**修复**：`git rm -r --cached .workbuddy`（保留本地文件、仅移出版本控制）+
`.gitignore` 追加 `.workbuddy/`。

**防回归**：新增任何工具目录时同步检查 `.gitignore`；
提交前 `git status --short` 目检一次清单。
