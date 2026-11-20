# 回收电池材料的取样争议与加工结算分析

本仓库包含合成业务资料、通用资料检查器（`sourcepack`）和业务分析应用（`recsettle`）。
字段、单位和业务记录的含义见 docs/资料口径.md，完整性目录为 data/catalog.json，
模型、口径与假设见 docs/方法说明.md。不得把文件自检成功解释成业务结论成立。

## 环境

Python 3.11.2；分析依赖已在 requirements.txt 固定版本。

    python3 -m pip install -r requirements.txt
    python3 -m sourcepack          # 资料完整性检查（不做业务解释）

## 业务应用 recsettle

围绕同一批实物，为采购、实验室和生产提供估值、加工、结算、复检规划与审计：

    python3 -m recsettle overview                     # 各批估值与状态总览
    python3 -m recsettle estimate --lot L6            # 单批：干基/可结算量、不确定性贡献
    python3 -m recsettle settle --lot L15             # 单批结算单：应付区间、暂付抵扣、追偿
    python3 -m recsettle process --save-plan plan.json  # 加工方案比较并保存推荐计划
    python3 -m recsettle plan-recheck --budget 1000   # 预算与期限下选择实际取样
    python3 -m recsettle audit --lot L6               # 差额追溯与可调整性
    python3 -m recsettle check --plan plan.json       # 一致性检查（含计划校验）

通用选项（置于子命令之后）：`--as-of` 历史截点（ISO 时刻，只用该时刻已可知的证据）、
`--sims` 蒙特卡洛次数、`--seed` 随机种子、`--json` 机器可读输出、`--data` 资料根目录。

要点：干基按容器净重加权；未检出按区间处理不为零；留样复测不增加容器代表性；
超历史配料范围的收益只作情景；污染物上界不能用金属收益抵消；已付款只抵扣不重付；
已投实物不可逆转，余料才可改配。

## 检查与测试

    python3 -m unittest discover -s tests -v
    python3 -m compileall -q sourcepack recsettle tests

测试覆盖资料容器、输入诊断与业务行为（容器加权、未检出区间、贡献分解、复检证据差异、
结算公式、硬边界、计划约束、差额追溯）。运行资料检查与测试不需要访问外部业务系统。
