import pandas as pd

# 专门开了个文件对 data frames 做研究
"""
DataFrame: 简单来说就是一个 有行名 + 列名的 '数据表格' (和Excel, SQL表一样), 只不过在Python中可以直接编程"控它"

一个dataframe数据示例:
name  age  score
0   Alice   18     90
1     Bob   21     85
2   Cathy   20     88
"""


## 创建一个dataframe
"""
一般使用'字典'进行创建
pd.DataFrame({'字段1': ['数据1', '数据2', ...], 
              '字段2': ['数据1', '数据2', ...],
              ...})

"""
df1=pd.DataFrame({'column_1':['b','b','a','c','a','a','b','e'],
                  'column_2':[2,2,1,3,1,1,2,5]})
print(df1);
print();

df2=pd.DataFrame({'column_1':['a','b','d'],
                  'column_4':['A','B','D']})
print(df2);
print("=======================")


## 将两个dataframe数据进行"内连接" & (和SQL中的内连接 & 外连接一个道理)
"""
内连接:
注: 以下三种方法均会返回一个新的 dataframe对象

多种不同的'内连接方式'
pd.merge(df对象a, df对象b, on="两个共有的列(字段)名", how="inner");

如果两边的'字段名不一致'(但是里面的值又对得上), 则可以用
pd.merge(df对象a, df对象b, left_on="对象a的字段名", right_on="对象b的字段名");

直接让python自己查两个df对象的'共有字段名' & 进行"内连接"
(注: 当两边的对象'没有共同字段名'时会直接报错!)
pd.merge(df对象a, df对象b)

外连接:
分三种情况, 两边全保留, 仅保留左边 和 仅保留右边
pd.merge(df对象a, df对象b, on="共有字段名", how="outer"); # 两边全部保留 (通常数据更多的那边可能会'吃亏'[缺数据])
pd.merge(df对象a, df对象b, on="共有字段名", how="left"); # 保留'df对象a' (留全左边的所有数据, 右边找不到'共同key映射'的为NaN)
pd.merge(df对象a, df对象b, on="共有字段名", how="right"); # 保留'df对象b' (留全右边的所有数据, 左边找不到'共同key映射'的为NaN)
"""
# 执行'内连接'
print(pd.merge(df1, df2)); # 以两边共有的'column_1'做内连接
print()

# 执行'外连接'
print(pd.merge(df1, df2, on="column_1", how="outer"));
print()
print(pd.merge(df1, df2, on="column_1", how="left")); # 留全 1 2 3 4 5 (从上往下按df1的顺序排列), 'b b a c...'
print();
print(pd.merge(df1, df2, on="column_1", how="right")); # 留全A B D (从上往下按df2的顺序排列, 'A B D')