import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# 这节的内容主要在研究如何用matplotlib画出各种'很高级'的图, 画图的目的主要可以分成以下几点
"""
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

# 设置一些'图形参数' (略)
large = 22; med = 16; small = 12
params = {
    'axes.titlesize': large,     # 子图标题字号
    'legend.fontsize': med,      # 图例字号
    'figure.figsize': (16, 10),  # 默认画布大小
    'axes.labelsize': med,       # 轴标签字号
    'axes.titlesize': med,
    'xtick.labelsize': med,      # X 轴刻度字号
    'ytick.labelsize': med,
    'figure.titlesize': large,   # 总标题字号
}
plt.rcParams.update(params)              # rcParams = 全局默认样式，一次设置全程生效
plt.style.use("seaborn-v0_8-whitegrid")  # 套用自带主题（白底+网格）

## 2.1 相关性 Correlation
### 散点图 Scatter Plot
# 导入'midwest_filter.csv'
# midwest = pd.read_csv("./datasets/midwest_filter.csv");
# categories = np.unique(midwest['category']) # 定义'有哪些分组' (利用数据集中的'category'字段进行分组)
# colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))] # 为每种分组设定一个颜色

# # 准备画出每个分组
# plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')

# # 循环 + 分组上色
# for i, category in enumerate(categories):
#     plt.scatter('area', 'poptotal',
#                 data=midwest.loc[midwest.category==category, :], 
#                 s=20, color=colors[i], label=str(category))

# # 设定一些小小的装饰
# plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000),
#               xlabel='Area', ylabel='Population')
# plt.xticks(fontsize=12); plt.yticks(fontsize=12)
# plt.title("Scatterplot of Midwest Area vs Population", fontsize=22)
# plt.legend(fontsize=12)

# plt.show()

### 热力图 heat map
# df = pd.read_csv("./datasets/mtcars.csv")
# df_numeric = df.select_dtypes(include='number')  # 只挑'数值列'，'文字列'算不了相关系数

# # 正式画图部分
# plt.figure(figsize=(12,10), dpi=80)
# sns.heatmap(df_numeric.corr(),   # corr() 算相关系数矩阵，取值 -1 ~ 1
#             xticklabels=df_numeric.columns,
#             yticklabels=df_numeric.columns,
#             cmap='RdYlGn',  # 红-黄-绿配色
#             center=0,    # 让 0 处在颜色正中
#             annot=True)  # 在格子里写出具体数字

# # 正式装饰 & 画图
# plt.title('Correlogram of mtcars', fontsize=22)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()

## 2.2 偏差 Deviation

# (放弃看tut给的抽象例子了, 直接参考Agent.py吧)