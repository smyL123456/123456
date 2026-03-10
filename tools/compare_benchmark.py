"""
compare_benchmark.py
====================
对比三分支与双分支在 AIGCBenchmark 上的评测结果。

用法：
    python tools/compare_benchmark.py \
        --csv3 results/eval_3branch/checkpoint-best.pth_AIGCBenchmark.csv \
        --csv2 results/eval_2branch/official_2branch.pth_AIGCBenchmark.csv

输出：
    - 控制台：逐 generator 对比表格 + 平均指标
    - 文件：results/comparison.csv（可选，--save 时输出）
"""

import argparse
import csv
import os
import sys


def load_csv(path):
    """读取 main_finetune.py 输出的评测 CSV，返回 {generator: (acc, ap)} 字典。"""
    if not os.path.exists(path):
        sys.exit(f"[错误] 找不到文件: {path}")

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    # 格式：第 0 行是 checkpoint 信息，第 1 行是表头，第 2 行起是数据
    if len(rows) < 3:
        sys.exit(f"[错误] CSV 行数不足，请检查文件: {path}")

    data = {}
    for row in rows[2:]:
        if len(row) < 3:
            continue
        generator = row[0].strip()
        try:
            acc = float(row[1])
            ap  = float(row[2])
        except ValueError:
            continue
        data[generator] = (acc, ap)
    return data


def main():
    parser = argparse.ArgumentParser(description="对比三分支与双分支 AIGCBenchmark 结果")
    parser.add_argument("--csv3", required=True, help="三分支评测 CSV 路径")
    parser.add_argument("--csv2", required=True, help="双分支评测 CSV 路径")
    parser.add_argument("--save",  default=None,  help="保存对比结果到此路径（可选）")
    args = parser.parse_args()

    d3 = load_csv(args.csv3)
    d2 = load_csv(args.csv2)

    # 取两者共有的 generator
    generators = [g for g in d3 if g in d2]
    if not generators:
        sys.exit("[错误] 两个 CSV 中没有共同的 generator，无法对比")

    # ---- 打印表格 ----
    header = f"{'Generator':<30} {'ACC-3b':>8} {'ACC-2b':>8} {'ΔACC':>8}  {'AP-3b':>8} {'AP-2b':>8} {'ΔAP':>8}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    acc_deltas = []
    ap_deltas  = []
    output_rows = [["generator", "acc_3b", "acc_2b", "delta_acc", "ap_3b", "ap_2b", "delta_ap"]]

    for g in generators:
        acc3, ap3 = d3[g]
        acc2, ap2 = d2[g]
        da = acc3 - acc2
        dp = ap3  - ap2
        acc_deltas.append(da)
        ap_deltas.append(dp)

        sign_a = "+" if da >= 0 else ""
        sign_p = "+" if dp >= 0 else ""
        print(f"{g:<30} {acc3:>8.4f} {acc2:>8.4f} {sign_a}{da:>7.4f}  {ap3:>8.4f} {ap2:>8.4f} {sign_p}{dp:>7.4f}")
        output_rows.append([g, acc3, acc2, da, ap3, ap2, dp])

    print(sep)
    mean_da = sum(acc_deltas) / len(acc_deltas)
    mean_dp = sum(ap_deltas)  / len(ap_deltas)
    sign_ma = "+" if mean_da >= 0 else ""
    sign_mp = "+" if mean_dp >= 0 else ""
    print(f"{'MEAN':<30} {'':>8} {'':>8} {sign_ma}{mean_da:>7.4f}  {'':>8} {'':>8} {sign_mp}{mean_dp:>7.4f}")
    print(sep)
    print(f"\n三分支 vs 双分支  平均 ACC delta: {sign_ma}{mean_da:.4f}   平均 AP delta: {sign_mp}{mean_dp:.4f}")

    # ---- 可选保存 ----
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(output_rows)
        print(f"\n对比结果已保存至: {args.save}")


if __name__ == "__main__":
    main()
