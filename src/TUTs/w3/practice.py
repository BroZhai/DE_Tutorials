import pandas as pd
a = [[1, 2, 3, 4, ],
     [2, 3, 4, 5, 6],
     [3, 4, 5, 6, 7]]
b = [[10, 20, 30, 40, 50],
     [30, 50, 60, 65, 70],
     [50, 60, 70, 75, 90],
     [60 ,70, 80, 90, 100]]
data = pd.DataFrame(a);
db = pd.DataFrame(b)
# print(db);
# print(data);
# 注: 上面创建的dataframe 没有行名和列名, pandas会直接用'行 / 列数'表示 (均从'0'开始)

# 取得均一化(0-1区间分布)的数值, 照搬公式即可 (数据中'每一列'的最大值 - 最小值 = 范围, (当前值 - 列最小值)/范围)
data = (data - data.min()) / (data.max() - data.min())
db = (db - db.min()) / (db.max() - db.min()) 
print(db);

# 取得标准化 standardization
db = pd.DataFrame(b);
db = (db - db.mean()) / db.std();
print(db);