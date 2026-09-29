# -*- coding: utf-8 -*-
"""
=================================
主题：Data Preprocessing (Pandas) + Regular Expression (re)

本脚本把教程里出现的每一个代码段落都翻译成"可直接运行的演示"：
  Part 0  环境准备与依赖安装（pip）
  Part 1  导入库 (import pandas as pd / import re)
  Part 2  数据预处理 (Data Preprocessing)
            2.1 数据清洗 Data Cleaning
                  - 基本操作 Basic Operations
                  - 处理缺失值 NAN
                  - 异常值检查 Unreasonable Data
                  - 重复值检查 Replicated Data
                  - 类型约束 & 列改名 Constrain Data Type
                  - 保存文件 Save Files
            2.2 数据集成 Data Integration  (merge)
            2.3 数据变换 Data Transformation (字符串 / 数值)
  Part 3  正则表达式 Regular Expression
  Part 4  练习答案 Practice (归一化 / 标准化 / 提取日期)

"""

import sys

# =========================================================
# Part 0  环境准备
# =========================================================
# 教程里写的是 Jupyter 的魔法命令  !pip install pandas
# 在普通 .py 脚本里不能这么写，需要用下面这种"在脚本里自己装包"的写法
# （实际项目中建议在命令行里 pip install，这里仅为演示教程对应代码）
def _ensure_package(pkg_name):
    try:
        __import__(pkg_name)
    except ImportError:
        import subprocess
        print(f"[setup] {pkg_name} 未安装，正在尝试 pip install {pkg_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

_ensure_package("pandas")

# =========================================================
# Part 1  导入库
# =========================================================
import pandas as pd      # 数据处理的主力库，惯例别名为 pd
import numpy as np       # 配合 pandas 做缺失值/数值运算


# =========================================================
# 构造演示数据（替代教程里的 movie_metadata.csv）
# 28 列全部字段保留，5043 行缩减为 12 行以便打印
# =========================================================
def build_movie_df():
    rows = [
        # color, director_name, num_critic_for_reviews, duration, director_facebook_likes,
        # actor_3_facebook_likes, actor_2_name, actor_1_facebook_likes, gross, genres,
        # num_user_for_reviews, language, country, content_rating, budget, title_year,
        # actor_2_facebook_likes, imdb_score, aspect_ratio, movie_facebook_likes
        ("Color", "James Cameron",   723.0, 178.0,    0.0,    855.0, "Joel David Moore", 1000.0, 760505847.0, "Action|Adventure|Fantasy|Sci-Fi", 3054.0, "English", "USA", "PG-13", 237000000.0, 2009.0, 936.0, 7.9, 1.78, 33000),
        ("Color", "Gore Verbinski",  302.0, 169.0,  563.0,   1000.0, "Orlando Bloom",    40000.0, 309404152.0, "Action|Adventure|Fantasy",       1238.0, "English", "USA", "PG-13", 300000000.0, 2007.0, 5000.0, 7.1, 2.35, 0),
        ("Color", "Sam Mendes",      602.0, 148.0,    0.0,    161.0, "Rory Kinnear",     11000.0, 200074175.0, "Action|Adventure|Thriller",       994.0, "English", "UK",  "PG-13", 245000000.0, 2015.0, 393.0, 6.8, 2.35, 85000),
        ("Color", "Christopher Nolan",813.0, 164.0,22000.0,  23000.0, "Christian Bale",  27000.0, 448130642.0, "Action|Thriller",                2701.0, "English", "USA", "PG-13", 250000000.0, 2012.0, 23000.0, 8.5, 2.35, 164000),
        (None,     "Doug Walker",      None,   None,  131.0,     None, "Rob Walker",        131.0,         None, "Documentary",                      None,      None, None,   None,        None,    None, 12.0, 7.1, None, 0),
        ("Color", "Andrew Stanton",  462.0, 132.0,  475.0,    530.0, "Samantha Morton",   640.0,  73058679.0, "Action|Adventure|Sci-Fi",          738.0, "English", "USA", "PG-13", 263700000.0, 2012.0, 632.0, 6.6, 2.35, 24000),
        ("Color", "Peter Jackson",   446.0, 201.0,    0.0,     84.0, "Thomas Kretschmann",6000.0, 218051260.0, "Action|Adventure|Drama|Romance",  2618.0, "English", "New Zealand", "PG-13", 207000000.0, 2005.0, 918.0, 7.2, 2.35, 0),
        ("Black and White","Akira Kurosawa",153.0,202.0,0.0,   4.0, "Minoru Chiaki",     304.0,   269061.0, "Action|Adventure|Drama",             596.0, "Japanese","Japan","Unrated", 2000000.0, 1954.0, 8.0, 8.7, 1.37, 11000),
        ("Color", "Scott Smith",       1.0,  87.0,    2.0,    318.0, "Daphne Zuniga",     637.0,         None, "Comedy|Drama",                       6.0, "English", "Canada","TV-14",None,         2013.0, 470.0, 7.7, None, 84),
        ("Color", None,               43.0,  43.0,     None,  319.0, "Valorie Curry",      841.0,         None, "Crime|Drama|Mystery|Thriller",     359.0, "English", "USA", "TV-14", None,         None,   593.0, 7.5, 16.00, 32000),
        ("Color", "Benjamin Roberds", 13.0,  76.0,    0.0,      0.0, "Maxwell Moody",       0.0,         None, "Drama|Horror|Thriller",              3.0, "English", "USA", None,    1400.0,      2013.0, 0.0, 6.3, None, 16),
        ("Color", "Jon Gunn",         43.0,  90.0,   16.0,     16.0, "Brian Herzlinger",    86.0,    85222.0, "Documentary",                         84.0, "English", "USA", "PG",   1100.0,      2004.0, 23.0, 6.6, 1.85, 456),
    ]
    cols = ["color","director_name","num_critic_for_reviews","duration","director_facebook_likes",
            "actor_3_facebook_likes","actor_2_name","actor_1_facebook_likes","gross","genres",
            "num_user_for_reviews","language","country","content_rating","budget","title_year",
            "actor_2_facebook_likes","imdb_score","aspect_ratio","movie_facebook_likes"]
    df = pd.DataFrame(rows, columns=cols)
    # 加一个纯英文电影标题列，供字符串变换示例使用
    df["movie_title"] = ["Avatar","Pirates","Skyfall","The Dark Knight","Nostalgia Critic","John Carter",
                         "King Kong","Seven Samurai","Comedy Demo","Crime Demo","Horror Demo","Doc Demo"]
    return df

# 教程原句：data = pd.read_csv(r'movie_metadata.csv', encoding="utf-8")
# 这里用 pd.DataFrame 直接构造，效果等价
data = build_movie_df()
print("\n[载入数据] 已构造示例电影数据集，shape =", data.shape)


# =========================================================
# Part 2  数据预处理
# =========================================================
print("\n" + "=" * 60)
print("Part 2.1  数据清洗 —— 基本操作 (Basic Operations)")
print("=" * 60)

# In[3] 看前 5 行：data.head()
print("\n--- data.head() : 看前 5 行，快速检查列名和数据长相 ---")
print(data.head())

# In[5] 看后 5 行：data.tail()
print("\n--- data.tail() : 看后 5 行，检查数据结尾是否完整 ---")
print(data.tail())

# In[4] 统计摘要：data.duration.describe()
print("\n--- data['duration'].describe() : 数值列的统计摘要 ---")
print(data["duration"].describe())
# 解释：count 是非空个数，mean/std 是均值标准差，min/25%/50%/75%/max 是分位数

# In[6] 选一列：data['color']
print("\n--- data['color'] : 选取单列，返回 Series ---")
print(data["color"])

# In[7] 取前 K 行：data['color'][:K]
K = 5
print(f"\n--- data['color'][:{K}] : 只取前 {K} 行 ---")
print(data["color"][:K])

# In[8] 选多列：data[["color","director_name"]]
print("\n--- data[['color','director_name']] : 选取多列，返回 DataFrame ---")
print(data[["color", "director_name"]])

# In[9] 条件过滤：data[data['duration'] > 150]
print("\n--- data[data['duration'] > 150] : 时长超过 150 分钟的电影 ---")
print(data[data["duration"] > 150][["movie_title", "duration", "director_name"]])


# ---------------------------------------------------------
# 2.1.2  处理缺失值 (Process NAN Data)
#   三种思路：1) 填充  2) 删行  3) 删列
# ---------------------------------------------------------
print("\n" + "-" * 60)
print("Part 2.1.2  处理缺失值")
print("-" * 60)

# In[10] data.isna() ：逐单元格判断是否为 NaN（返回同形状的布尔表）
print("\n--- data.isna() 概览（每列缺失个数） ---")
print(data.isna().sum())

# In[11] 填充缺失值
#   用固定值填：data.country.fillna("baka")
#   用统计量填：data.duration.fillna(data.duration.mean())
data["country"] = data["country"].fillna("baka")
data["duration"] = data["duration"].fillna(data["duration"].mean())
print("\n--- 填充后，'country' 与 'duration' 不再有缺失 ---")
print("duration NaN 数:", data["duration"].isna().sum(),
      "| country NaN 数:", data["country"].isna().sum())

# In[12] data.dropna() ：删掉含 NaN 的行（默认 any，即只要有一列为空就删）
print("\n--- data.dropna() : 删掉任意列有缺失的行 ---")
print("删除后 shape:", data.dropna().shape)

# 教程里给的其他变体：
print("\n--- 其他 dropna 用法演示 ---")
print("dropna(how='all')  只删全为空的行  ->", data.dropna(how="all").shape)
print("dropna(thresh=25)  至少 25 列非空   ->", data.dropna(thresh=25).shape)
print("dropna(axis=1, how='all') 删全空的列 ->", data.dropna(axis=1, how="all").shape)
# dropna(axis=1, how='any') 会删掉所有含缺失的列，演示数据几乎全删，这里只展示不执行
# print(data.dropna(axis=1, how='any').shape)


# ---------------------------------------------------------
# 2.1.3  异常值检查 (Unreasonable Data)
#   思路：用业务逻辑去反查不合法的值
# ---------------------------------------------------------
print("\n" + "-" * 60)
print("Part 2.1.3  异常值检查")
print("-" * 60)

# 年份不应晚于数据收集年份
print("\n--- 年份大于 2015 的记录（可能是录入错误） ---")
print(data[data["title_year"] > 2015][["movie_title", "title_year"]])

# IMDb 评分上限为 10
print("\n--- IMDb 评分大于 10 的记录（不可能，属于异常） ---")
print(data[data["imdb_score"] > 10][["movie_title", "imdb_score"]])


# ---------------------------------------------------------
# 2.1.4  重复值检查 (Replicated Data)
# ---------------------------------------------------------
print("\n" + "-" * 60)
print("Part 2.1.4  重复值检查")
print("-" * 60)

# 教程原数据集没有重复行，所以用它给的 df 示例
df_dup = pd.DataFrame({
    "brand":  ["Yum Yum", "Yum Yum", "Indomie", "Indomie", "Indomie"],
    "style":  ["cup", "cup", "cup", "pack", "pack"],
    "rating": [4, 4, 3.5, 15, 5],
})
print("\n--- 原始重复示例 df ---")
print(df_dup)

print("\n--- df.duplicated() 默认 keep='first'：首行标记 False，其后重复行 True ---")
print(df_dup.duplicated())

print("\n--- keep='last'：保留最后一行，前面的重复行标 True ---")
print(df_dup.duplicated(keep="last"))

print("\n--- keep=False：所有重复行都标 True ---")
print(df_dup.duplicated(keep=False))

print("\n--- subset=['brand']：只看 brand 列判定重复 ---")
print(df_dup.duplicated(subset=["brand"]))

# 实际去重：drop_duplicates()
print("\n--- df_dup.drop_duplicates() : 去掉完全重复的行 ---")
print(df_dup.drop_duplicates())


# ---------------------------------------------------------
# 2.1.5  类型约束 & 列改名 (Constrain Data Type)
# ---------------------------------------------------------
print("\n" + "-" * 60)
print("Part 2.1.5  类型约束 & 列改名")
print("- * 为避免破坏上面步骤，这里在副本上演示 *")
print("-" * 60)

# 读数据时指定列类型
# data = pd.read_csv(r'movie_metadata.csv', dtype={'num_voted_users': int, "title_year": str})
data2 = data.copy()
print("\n--- 读入后把 title_year 转成字符串类型 ---")
data2["title_year"] = data2["title_year"].astype(str)
print("title_year dtype:", data2["title_year"].dtype)

# 重命名列，便于人理解
data2 = data2.rename(columns={
    "title_year": "release_year",
    "movie_facebook_likes": "facebook_likes",
})
print("\n--- 重命名后列头 ---")
print(data2.columns.tolist())


# ---------------------------------------------------------
# 2.1.6  保存文件 (Save Files)
# ---------------------------------------------------------
print("\n--- data.to_csv('cleanfile.csv', encoding='utf-8') ---")
data2.to_csv("cleanfile.csv", encoding="utf-8", index=False)
print("[保存] 已写出 cleanfile.csv（index=False 表示不把行号写进文件）")


# ---------------------------------------------------------
# 2.2  数据集成 (Data Integration) —— pd.merge
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("Part 2.2  数据集成 —— pd.merge")
print("=" * 60)

df1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "a", "b"], "data1": range(7)})
df2 = pd.DataFrame({"key": ["a", "b", "d"], "data2": range(3)})
print("\n--- df1 / df2 ---")
print(df1); print(df2)

print("\n--- pd.merge(df1, df2) : 默认按共同列 'key' 内连接 ---")
print(pd.merge(df1, df2))

print("\n--- 显式指定 on='key' ---")
print(pd.merge(df1, df2, on="key"))

# 列名不同时用 left_on / right_on
df3 = pd.DataFrame({"lkey": ["b", "b", "a", "c", "a", "a", "b"], "data1": range(7)})
df4 = pd.DataFrame({"rkey": ["a", "b", "d"], "data2": range(3)})
print("\n--- 列名不同时用 left_on / right_on ---")
print(pd.merge(df3, df4, left_on="lkey", right_on="rkey"))

print("\n--- how='outer' 外连接：保留左右两边所有键 ---")
print(pd.merge(df1, df2, on="key", how="outer"))


# ---------------------------------------------------------
# 2.3  数据变换 (Data Transformation)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("Part 2.3  数据变换")
print("=" * 60)

# ---- 字符串变换 ----
print("\n--- 字符串：转大写 / 转小写 / 去首尾空白 ---")
print("upper:", data["director_name"].dropna().str.upper().head(3).tolist())
print("lower:", data["director_name"].dropna().str.lower().head(3).tolist())
print("strip:", data["movie_title"].str.strip().head(3).tolist())

# ---- 数值变换 ----
print("\n--- 数值：单位换算（分钟 -> 小时） ---")
print((data["duration"] / 60.0).head())

print("\n--- 归一化 (min-max) ---")
dur = data["duration"]
norm_dur = (dur - dur.min()) / (dur.max() - dur.min())
print(norm_dur.round(3))

print("\n--- 标准化 (z-score) ---")
std_dur = (dur - dur.mean()) / dur.std()
print(std_dur.round(3))

print("\n--- 离散化 pd.qcut 分 5 档 ---")
qcut = pd.qcut(data["duration"], 5)
print(qcut.value_counts().sort_index())



# =========================================================
# Part 4  练习答案 (Practice)
# =========================================================
print("\n" + "=" * 60)
print("Part 4  练习答案")
print("=" * 60)

# ---- 数据预处理练习：归一化 & 标准化 ----
a = [[1, 2, 3, 4],
     [2, 3, 4, 5, 6],
     [3, 4, 5, 6, 7]]
prac = pd.DataFrame(a)
print("\n原始数据:\n", prac)

# 需要先把不等长的行补齐（否则没法按列 min-max），这里用 NaN 对齐
prac = pd.DataFrame(prac).apply(lambda c: pd.to_numeric(c, errors="coerce"))
print("\n各列最小/最大值:\n", prac.min(), "\n", prac.max())

# 归一化：按列做 (x - min) / (max - min)
norm_data = (prac - prac.min()) / (prac.max() - prac.min())
print("\n归一化结果:\n", norm_data.round(3))

# 标准化：按列做 (x - mean) / std
std_data = (prac - prac.mean()) / prac.std()
print("\n标准化结果:\n", std_data.round(3))

# ---- 正则练习：提取日期（三种分隔符 / . -）----
print("\n[正则练习] 提取日期")
pattern = re.compile(r"\d{4}[/\-.]\d{2}[/\-.]\d{2}")
strs = "Today is 2022/09/13, today in the last year is 2021.09.13, today in the next year is 2023-09-13"
result = pattern.findall(strs)
print("匹配结果 ->", result)

print("\n" + "=" * 60)
print("全部演示结束")
print("=" * 60)
