# -*- coding: utf-8 -*-
"""
这节的内容主要在研究如何用matplotlib画出各种'很高级'的图, 画图的目的主要可以分成以下几点

2.1 相关性 Correlation: 当变量A变化时, 另一个变量B会怎么变动? (变量A, B之间有什么关系?)
    -> 2.1.1 散点图 / 2.1.2 相关图(热力图)

2.2 偏差   Deviation: 计算出'数据均值', 看每个值和'平均值'差多少 (z-score 标准化就是这样类似的思想, 计算每个数值与准直有多少个'标准差')
    -> 2.2.1 发散条形 / 2.2.2 发散文本 / 2.2.3 面积图

2.3 排序   Ranking: 将多个数据按'某个指标'排序之后, 看看'相邻之间差多少'
    -> 2.3.1 棒棒糖图 / 2.3.2 点图

2.4 分布   Distribution: 看'数据整体'长什么样, '分布/集中区间'是怎样的, 有没有'异常值' (outliers)
    -> 2.4.1 直方图 / 2.4.2 密度图 /2.4.3 密度+直方图 / 2.4.4 箱线图

2.5 构成   Composition: 
    -> 2.5.1 饼图 / 2.5.2 条形图
"""

import os
import matplotlib
# matplotlib.use("Agg")            # 无显示器环境必须加这行
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# =====================================================
# Part 1  全局样式（教程原代码，一次设置，后面所有图都生效）
# =====================================================
large = 22; med = 16; small = 12
params = {
    'axes.titlesize': large,        # 图的标题字号
    'legend.fontsize': med,         # 图例字号
    'figure.figsize': (16, 10),     # 默认画布大小
    'axes.labelsize': med,          # 坐标轴名字字号
    'xtick.labelsize': med,         # X 轴刻度字号
    'ytick.labelsize': med,         # Y 轴刻度字号
    'figure.titlesize': large,      # 总标题字号
}
# 注：教程 PDF 里把 xtick.labelsize 印成了 xticklabelsize（少了点），
#     rcParams 只认带点的写法，写错了会直接报 KeyError。
plt.rcParams.update(params)
plt.style.use("seaborn-v0_8-whitegrid")   # 白底 + 网格主题
# 注意：教程写的 "seaborn-whitegrid" 是旧名字，新版 matplotlib 要加 v0_8 前缀

OUT = "output_plots"
os.makedirs(OUT, exist_ok=True)

def save(name):
    """保存当前这张图，然后清空画布，避免下一张图叠上来"""
    plt.savefig(os.path.join(OUT, name), dpi=80, bbox_inches="tight")
    plt.close()
    print("  已保存 ->", name)


# =====================================================
# 准备数据：只有 6 行，一眼看得完
# =====================================================
df = pd.DataFrame({
    "name":        ["Ann", "Bob", "Cid", "Dan", "Eve", "Fay"],
    "study_hours": [2, 4, 6, 8, 10, 12],        # 每周学习小时
    "score":       [55, 62, 70, 78, 85, 92],    # 考试分数（和学习时间正相关）
    "sleep_hours": [8.0, 7.5, 7.0, 6.5, 6.0, 5.5],
    "gender":      ["F", "M", "M", "M", "F", "F"],
})
df.to_csv("./datasets/students.csv", encoding="utf-8");

# 另一份数据：20 个学生的分数，用来画"分布"类的图（6 行太少画不出分布）
scores = pd.DataFrame({
    "score": [55, 60, 62, 65, 65, 68, 70, 70, 72, 75,
              75, 78, 80, 80, 82, 85, 88, 90, 92, 95],
    "gender": ["F", "M"] * 10,
})
scores.to_csv("./datasets/students_scores.csv", encoding="utf-8");

# 第三份数据：6 个月销量，用来画"偏差"类的面积图
months = pd.DataFrame({
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "sales": [100, 120, 110, 140, 130, 150],
})
months.to_csv("./datasets/months_sales.csv", encoding="utf-8");

# 第四份数据：5 个水果的销量，用来画"排序"类的图
fruits = pd.DataFrame({
    "fruit": ["Apple", "Banana", "Cherry", "Durian", "Grape"],
    "sales": [50, 40, 30, 20, 10],
})
fruits.to_csv("./datasets/fruits_sales.csv", encoding="utf-8");

# print("示例数据 df：")
# print(df)


# =====================================================
# 2.1 相关性 Correlation —— 两个变量之间有关系吗？
# =====================================================

# ---------- 2.1.1 散点图 Scatter plot ----------
# 每个点 = 一个学生；横轴学习时长，纵轴分数。点越像一条斜线，关系越强。
print("\n[2.1.1] 散点图：学习时长 vs 分数")
plt.figure(figsize=(8, 5))
plt.scatter(df["study_hours"], df["score"], s=100, color="steelblue")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Scatter: Study Hours vs Score")
save("01_scatter.png")

# 进阶版：按性别分组上色（教程里的做法，用循环逐个分组画）
plt.figure(figsize=(8, 5))
colors = {"F": "hotpink", "M": "steelblue"}
for g in df["gender"].unique():                    # 遍历每个分组
    part = df[df["gender"] == g]                   # 取出该组的数据（注意是 == 不是 =）
    plt.scatter(part["study_hours"], part["score"],
                s=100, color=colors[g], label=g)   # label 用于生成图例
plt.xlabel("Study Hours"); plt.ylabel("Score")
plt.title("Scatter Colored by Gender")
plt.legend(title="Gender")                         # 教程写的 plt.Legend(...) 是笔误，正确是 legend()
save("01b_scatter_group.png")


# ---------- 2.1.2 相关图（热力图）Correlogram ----------
# 一次性算出"所有数值列两两之间的相关系数"，用颜色表示：绿=正相关，红=负相关
print("[2.1.2] 相关图：所有数值列的相关系数矩阵")
num = df.select_dtypes(include="number")   # 只挑数值列，文字列算不了相关系数
plt.figure(figsize=(7, 6))
sns.heatmap(num.corr(),          # corr() 算相关系数矩阵，取值 -1 ~ 1
            annot=True,          # 在格子里写出具体数字
            cmap="RdYlGn",       # 红-黄-绿配色
            center=0)            # 让 0 处在颜色正中
plt.title("Correlogram")
save("02_correlogram.png")


# =====================================================
# 2.2 偏差 Deviation —— 谁偏离了平均水平？偏了多少？
# =====================================================

# ---------- 2.2.1 发散条形 Diverging Bars ----------
# 思路：把分数标准化成 z 分数（0 = 正好等于平均），然后从 0 画一根横向条
print("\n[2.2.1] 发散条形：每个学生偏离平均分多少")
d = df.copy()
d["z"] = (d["score"] - d["score"].mean()) / d["score"].std()   # z 分数：负数=低于平均
d["color"] = ["red" if v < 0 else "green" for v in d["z"]]     # 低于平均红，高于平均绿
d = d.sort_values("z").reset_index(drop=True)                  # 排序后图形更整齐

plt.figure(figsize=(8, 5))
plt.hlines(y=d.index,              # 每一根线所在的高度（第几个学生）
           xmin=0,                 # 从左边的 0 开始
           xmax=d["z"],            # 画到 z 值处（负数就往左画）
           color=d["color"], linewidth=8, alpha=0.6)
plt.yticks(d.index, d["name"])     # 把刻度数字换成学生名字
plt.axvline(0, color="black")      # 画一条 0 的参考线
plt.xlabel("Score (z-score)"); plt.title("Diverging Bars")
save("03_diverging_bars.png")


# ---------- 2.2.2 发散文本 Diverging Texts ----------
# 和上面完全一样，只是不画"条"，而是在末端直接写出数值
print("[2.2.2] 发散文本：把数值直接写在末端")
plt.figure(figsize=(8, 5))
plt.hlines(y=d.index, xmin=0, xmax=d["z"], color="gray", alpha=0.5)
for x, y in zip(d["z"], d.index):
    plt.text(x, y, round(x, 2),                             # 在 (x, y) 位置写字
             ha="right" if x < 0 else "left",               # 负数写左边，正数写右边
             va="center", fontsize=13,
             color="red" if x < 0 else "green")
plt.yticks(d.index, d["name"])
plt.axvline(0, color="black")
plt.xlim(-2, 2)
plt.title("Diverging Texts")
save("04_diverging_texts.png")


# ---------- 2.2.3 面积图 Area Chart ----------
# 用 fill_between 把曲线下方填色：涨的填绿，跌的填红，能看出"持续了多久"
print("[2.2.3] 面积图：月度销量的涨跌")
x = np.arange(len(months))                       # x = 0,1,2,3,4,5
growth = months["sales"].pct_change().fillna(0) * 100   # 环比增长率 %
plt.figure(figsize=(8, 5))
plt.fill_between(x, growth, 0, where=growth >= 0, facecolor="green", alpha=0.6)
plt.fill_between(x, growth, 0, where=growth < 0,  facecolor="red",   alpha=0.6)
# 教程原代码两次都写了 growth >= 0（第二个应为 < 0），是个 bug，这里已修正
plt.plot(x, growth, color="black")               # 再画一条黑线勾出轮廓
plt.xticks(x, months["month"])
plt.axhline(0, color="black")
plt.ylabel("Growth %"); plt.title("Area Chart")
save("05_area_chart.png")


# =====================================================
# 2.3 排序 Ranking —— 谁排第一？彼此差多少？
# =====================================================

# ---------- 2.3.1 棒棒糖图 Lollipop Chart ----------
# 一根竖线（糖棍）+ 一个圆点（糖头），比柱状图更清爽
print("\n[2.3.1] 棒棒糖图：水果销量排名")
f = fruits.sort_values("sales").reset_index(drop=True)   # 先排序
plt.figure(figsize=(8, 5))
plt.vlines(x=f.index, ymin=0, ymax=f["sales"], color="firebrick", linewidth=3)  # 棍
plt.scatter(f.index, f["sales"], s=150, color="firebrick", zorder=3)            # 头
plt.xticks(f.index, f["fruit"])
plt.ylabel("Sales"); plt.title("Lollipop Chart")
save("06_lollipop.png")


# ---------- 2.3.2 点图 Dot Plot ----------
# 同样是排名，但把点横向排列，更容易看出点与点之间的距离
print("[2.3.2] 点图：横向看排名差距")
plt.figure(figsize=(8, 5))
plt.hlines(y=f.index, xmin=0, xmax=60, color="gray", linestyles="dashdot")   # 灰色引导线
plt.scatter(f["sales"], f.index, s=150, color="firebrick", zorder=3)
plt.yticks(f.index, f["fruit"])
plt.xlabel("Sales"); plt.title("Dot Plot")
save("07_dot_plot.png")


# =====================================================
# 2.4 分布 Distribution —— 数据长什么样？集中在哪？
# =====================================================

# ---------- 2.4.1 直方图 Histogram ----------
# 把分数切成若干个区间，数每个区间里有多少人
print("\n[2.4.1] 直方图：分数分布")
plt.figure(figsize=(8, 5))
plt.hist(scores["score"], bins=8, color="steelblue", edgecolor="white")
plt.xlabel("Score"); plt.ylabel("Count"); plt.title("Histogram")
save("08_histogram.png")


# ---------- 2.4.2 密度图 Density Plot ----------
# 直方图的"平滑版"，像一条山丘曲线；多组叠加时比直方图清楚
print("[2.4.2] 密度图：男女分数分布对比")
plt.figure(figsize=(8, 5))
sns.kdeplot(scores[scores.gender == "F"]["score"], fill=True, color="hotpink",  label="F")
sns.kdeplot(scores[scores.gender == "M"]["score"], fill=True, color="steelblue", label="M")
plt.xlabel("Score"); plt.title("Density Plot")
plt.legend()
save("09_density.png")


# ---------- 2.4.3 直方图 + 密度曲线 ----------
# 两者合体：既有精确计数，又有平滑趋势
# 教程用的是 sns.distplot（已废弃），新版应写 sns.histplot(..., kde=True)
print("[2.4.3] 直方图 + 密度曲线")
plt.figure(figsize=(8, 5))
sns.histplot(scores["score"], bins=8, kde=True, color="steelblue")
plt.xlabel("Score"); plt.title("Histogram + Density")
save("10_hist_density.png")


# ---------- 2.4.4 箱线图 Box Plot ----------
# 一个箱子概括一组数据：中线=中位数，箱子=中间50%的范围，小点=离群值
print("[2.4.4] 箱线图：男女分数对比")
plt.figure(figsize=(8, 5))
sns.boxplot(x="gender", y="score", hue="gender", data=scores,
            palette="Set2", legend=False)
# 新版本 seaborn 要求：用了 palette 就要配上 hue，否则会报警告
plt.title("Box Plot")
save("11_boxplot.png")


# =====================================================
# 2.5 构成 Composition —— 整体由哪些部分组成？各占多少？
# =====================================================

# ---------- 2.5.1 饼图 Pie Chart ----------
# 扇形大小 = 占比。教程特别提醒：必须把百分比写出来，否则容易误读
print("\n[2.5.1] 饼图：男女比例")
counts = df["gender"].value_counts()          # 数一下 M、F 各有多少人
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=counts.index, autopct="%1.0f%%",   # autopct 自动标百分比
        colors=["hotpink", "steelblue"])
plt.title("Pie Chart")
save("12_pie.png")


# ---------- 2.5.2 条形图 Bar Chart ----------
# 用柱子高度表示数量，比饼图更容易比较大小
print("[2.5.2] 条形图：男女人数")
plt.figure(figsize=(7, 5))
plt.bar(counts.index, counts.values, color=["hotpink", "steelblue"], width=0.5)
for i, v in enumerate(counts.values):         # 在每根柱子顶上写出具体人数
    plt.text(i, v, v, ha="center", va="bottom", fontsize=14)
plt.ylabel("Count"); plt.ylim(0, 5); plt.title("Bar Chart")
save("13_bar.png")


# =====================================================
# Part 3  练习 Practice
# =====================================================

# ---------- 3.1 stripplot（抖动散点）----------
# 分类轴上的点很容易重叠，jitter=True 会给点加一点随机偏移把它们散开
print("\n[3.1] stripplot：抖动散点，避免点重叠")
plt.figure(figsize=(8, 5))
sns.stripplot(x="gender", y="score", data=scores, jitter=True, size=8, alpha=0.7)
plt.title("Stripplot (jitter)")
save("14_stripplot.png")


# ---------- 3.2 箱线图 + 散点 ----------
# 先画箱线图看分布，再叠加散点看样本量，两者共用一张图
print("[3.2] 箱线图 + 散点")
plt.figure(figsize=(8, 5))
sns.boxplot(x="gender", y="score", hue="gender", data=scores,
            boxprops={"facecolor": "white"},   # 白色箱体，避免盖住后面画的点
            showfliers=False, palette="Set2", legend=False)
sns.stripplot(x="gender", y="score", data=scores, color="black", size=6, alpha=0.6)
plt.title("Box Plot + Points")
save("15_box_dot.png")

print("\n全部完成！共 16 张图，都在 ./%s/ 目录下" % OUT)
