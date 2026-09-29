# 这堂内容研究pandas库是什么 & 基础用法
import pandas as pd

## 数据读取, 支持 csv, excel, json格式, 返回一个pandas.Series对象 (读取的数据对象)
"""
pd.read_csv("文件名/路径");
pd.read_excel("文件名/路径");
pd.read_json("文件名/路径");
如果是中文数据, 可以额外加个'字符编码'(utf-8), 通常为了保险, 我们都会带上一个字符编码
e.g., pd.reaa_csv("文件名/路径", encoding="utf-8");
"""
data_obj = pd.read_csv(r'minimized_data.csv', encoding="utf-8"); # 这里的data_obj对象即是一个'数据对象'
# data_types = pd.read_csv(r'minimized_data.csv', dtype={'num_voted_users': int, "title_year": str}) # 可以以这种方式'提前定义'字段类型
# data_renames = data_obj.rename(columns== {'title_year':'release_date', 'movie_facebook_likes':'facebook_likes'})) # 可以以这种方式对字段'重命名'

## 数据输出(保存)
"""
pd.to_csv("文件名/路径", encoding="...");
pd.to_excel("文件名/路径", encoding="...");
pd.to_json("文件名/路径", encoding="...");
"""

## pandas.Series对象的常用方法
"""
数据对象.head()        # 前 5 行
数据对象.tail()        # 后 5 行
数据对象.shape         # (行数, 列数)
数据对象.columns       # 列名
数据对象.dtypes        # 每列类型
数据对象.info()        # 结构信息
数据对象.describe()    # 针对'每一列'的统计：均值、最大最小等
    Tips: 我们也可以只选某个'特定字段'进行操作: 数据对象.某一字段.desc()
    数据对象.ziduanA.head() --> 仅显示'ziduanA'列的头五行信息
"""
# print(f"开头5行的数据:\n {data_obj.head()}");
print(f"头5个director_name: \n{data_obj.director_name.head()}"); # 也可以对'单独字段'用
# print(data_obj.describe());
print("=============================");

## 选取'特定数据列' / '多列': 
"""
选一列: 数据对象["字段名1"]... (可选[:]) 
    (Tips: 可以用Python数组的'节选[:] 和 [::]'方式来灵活控制数组)
选多列 (注: 这个要用'二维数组'): 数据对象[["字段名1", "字段名2", ...]]

按'位置'取 (注意这个'.loc'一定要带上)
数据对象.loc[行]: 取一行信息
    数据对象.loc[起始行:终止行]: 取'一定范围'的行信息
数据对象.loc[行, 列] / 数据对象.loc[行, '列字段名']: 
    数据对象.loc[起始行:终止行, 起始列数:终止列数] / 数据对象.loc[起始行:终止行, ['列字段名1', '列字段名2', ...]]: 取'一定范围'的行 & 列信息
"""
print(data_obj['director_name'][:5:2]); # 可以将data_obj['字段名'] 摘出来的东西想成一个'数组', 截取下标范围[0,5), 每隔两个元素取一个
print();
print(data_obj[['director_name','duration']][:5]);
print()
# 上述指令也可以用.loc位置来取 (相当于把外部的[:]搬到内部了, 但是字段选择可以用'字段名', 更灵活)
print(data_obj.loc[0:5, ['director_name', 'duration']]);
print("=============================");

## 简单数据判断 & 条件过滤 (用到上面'选行列'摘数据后, 再判断)
print("条件过滤");
filtered = data_obj[data_obj['duration'] > 160] # 这里内部先用条件判断'哪些行满足条件' --> 每行有个True False, 随后在外部用整体的'数据'来读每行的True/False 判断哪些留, 哪些不要 (进行过滤)
print(filtered.loc[0:5, ['director_name', 'duration']]);
print("=============================");

## 数据清理: 处理NaN(空值) Data Cleaning
"""
数据对象.isna().sum()              # 每列缺失数量
数据对象.dropna()                  # 删掉有缺失的行
    dropna(how="");                     # 定义'删除条件' how = 'any' ('任何一栏为NaN'就删整行) / how = 'all' ('整行为NaN'才删)
    dropna(thresh=数字)                 # 上方的一种'更精细'的定义: > 多少个NaN值就删 (e.g., thresh=2, 该行有'2个以上的NaN栏位'就删)
    dropna(axis=0/1, how / thresh)     # axis定义"删行还是删列", 默认axis=0(删除这一整行), axis=1则是删除'大表中的'这一整列''(慎用!!)
数据对象.fillna("填充内容")         # 指定缺失值的'填充内容' & 进行填充 
数据对象.duplicated()              # 检查有哪些重复数据 (默认 keep='first'：首行标记 False, 其后重复行 True)
    数据对象.duplicated(keep="last")        # keep="last", 将'首行'视为重复行, 只保留最后一行 (重复的最后一行为false, 其他为true)
    数据对象.duplicated(keep=False)         # 如果多行重复, 则全部标True (包括'首行')
    数据对象.duplicated(subset=['字段'])    # 不看整行, 只看'某个字段'是否有'重复值' (若字段重复, 对应的行也'整体被视为'重复)
上面的去重方法: 数据对象 = 数据对象[数据对象.duplicated(...)]
    有个'更直接'的方法:
    数据对象.drop_duplicates()         # 直接执行'去重' (判断依据为'整行')
    
"""
print("更改前:");
print(data_obj.loc[4:5, ['color', 'director_name']].isna()); # 看四-五行的'color' 和'director_name'字段哪些为空, 返回True/False
data_obj.loc[4:5, ['color', 'director_name']] = data_obj.loc[4:5, ['color', 'director_name']].fillna("114.5"); # 将'对应缺失段'的内容 用 '补好内容'的段覆盖
print("更改后, 但是还没有写入到文件");
print(data_obj.loc[4:5, ['color', 'director_name']].isna());
print()

# 用个独立的小例子来演示'drop_duplicated'
df = pd.DataFrame({
    'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
    'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
    'rating': [4, 4, 3.5, 15, 5]
}) # 第二行数据重复
# print(df);
# print(df.duplicated()); # 打印出确实是'第二行重复'
# df = df.drop_duplicates(); # 将去重后的结果重新赋给 df (原数据去重)
print(df.drop_duplicates());
print("=============================");

## 数据简单变换 Data Transformation
"""
("字符串"内容变换)
数据对象['指定字段名'].str.upper();  # 摘出来的'指定字段'内容(str)全大写
数据对象['指定字段名'].str.lower();  # 同上, 这里是全小写
数据对象['指定字段名'].str.strip();  # 去除内容中所有的'空白符' (\s \t 空格等...)

(数值 内容变换, 直接直接展示几个常用的'数学公式', 直接参考就行了)
这里以'data'来指代 "数据对象['指定字段名']"
min-max归一化(normalization): (data - data.min()) / (data.max() - data.min())
标准化(z-score): (data - data.mean() / data.std())
离散化(将数据分为不同'标签类型', 假设标签类别为5个): 
quct = pd.qcut(data, 5)
print(quct = qcut.value_counts().sort_index())
"""