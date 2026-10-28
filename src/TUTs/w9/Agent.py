# -*- coding: utf-8 -*-
"""
信息检索三模型极简演示：Boolean Retrieval / TF-IDF / BM25
对应 CS5481 Tutorial 9 的核心内容
"""

import numpy as np
from collections import Counter


# ============================================================
# 0. 准备语料（和教程里一样，先做简单的分词与清洗）
# ============================================================
docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends",
]

# 分词：按空格切分，去掉标点，统一小写（简单起见）
def tokenize(text):
    return [w.strip(".,!?").lower() for w in text.split()]

tokenized_docs = [tokenize(d) for d in docs]

# 构建排序后的词表，保证向量维度可复现
vocab = sorted({w for doc in tokenized_docs for w in doc})
word2id = {w: i for i, w in enumerate(vocab)}

print("汇总出的全局(独立)词表:", vocab)
print()


# ============================================================
# 1. 布尔检索 (Boolean Retrieval)
# ============================================================
# 构造 文档数 × 词表大小 的二值矩阵：词出现=1，否则=0
bool_matrix = np.zeros((len(docs), len(vocab)), dtype=int)
for i, doc in enumerate(tokenized_docs):
    for w in doc:
        bool_matrix[i, word2id[w]] = 1

print("全局词表中的词在每个'文档'中出现的Boolean值:")
print(bool_matrix)
print("形状:", bool_matrix.shape, "(行=文档, 列=词)")
print()

def boolean_query(terms, op="AND"):
    """用 NumPy 布尔运算实现 AND / OR 查询"""
    # 每个查询词对应一列布尔向量
    cols = [bool_matrix[:, word2id[t]] == 1 for t in terms if t in word2id]
    if not cols:
        return []
    if op == "AND":
        result = np.logical_and.reduce(cols)   # 所有词都出现
    else:  # OR
        result = np.logical_or.reduce(cols)    # 至少一个词出现
    # 返回命中文档的编号（从 1 开始，更符合直觉）
    return [i + 1 for i, hit in enumerate(result) if hit]

print("查询 'cat AND mat' ->", boolean_query(["cat", "mat"], "AND"))
print("查询 'cat OR dog'  ->", boolean_query(["cat", "dog"], "OR"))
print()


# ============================================================
# 2. 向量空间模型 (TF-IDF)
# ============================================================
N = len(docs)  # 文档总数

# 2.1 计算 TF：词频 / 文档长度
tf = np.zeros((N, len(vocab)))
for i, doc in enumerate(tokenized_docs):
    for w in doc:
        tf[i, word2id[w]] += 1 / len(doc)   # 归一化词频

# 2.2 计算 IDF：ln(N / (df + 1))，df = 包含该词的文档数
df = np.zeros(len(vocab))
for w in vocab:
    df[word2id[w]] = sum(1 for doc in tokenized_docs if w in doc)
idf = np.log(N / (df + 1))

# 2.3 TF-IDF 矩阵
tfidf = tf * idf

print("TF-IDF 矩阵 (每行一个文档):")
print(np.round(tfidf, 4))
print()

def tfidf_query_vector(query):
    """把查询也转成同样维度的 TF-IDF 向量"""
    toks = tokenize(query)
    q_tf = np.zeros(len(vocab))
    for w in toks:
        if w in word2id:                    # 忽略词表外的词
            q_tf[word2id[w]] += 1 / len(toks)
    return q_tf * idf

def cosine_sim(a, b):
    """余弦相似度，任一向量为零向量时返回 0"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

query = "cat dog"
q_vec = tfidf_query_vector(query)

# 点积得分 vs 余弦相似度得分
dot_scores = tfidf @ q_vec
cos_scores = np.array([cosine_sim(tfidf[i], q_vec) for i in range(N)])

print("查询:", query)
print("点积得分:     ", np.round(dot_scores, 4))
print("余弦相似度得分:", np.round(cos_scores, 4))
print("按余弦排序文档:", [i + 1 for i in np.argsort(-cos_scores)])
print()


# ============================================================
# 3. BM25 模型
# ============================================================
class BM25:
    """
    简化版 BM25：
    score = Σ IDF(q) * [ f*(k1+1) / (f + K) ] * [ qf*(k2+1) / (qf+k2) ]
    其中 K = k1 * (1 - b + b * dl / avg_dl)
    """
    def __init__(self, tokenized_docs, k1=1.5, k2=1.0, b=0.75):
        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.k1, self.k2, self.b = k1, k2, b
        self.avg_dl = np.mean([len(d) for d in tokenized_docs])  # 平均文档长度

        # 每个文档的词频统计
        self.freqs = [Counter(d) for d in tokenized_docs]

        # BM25 的 IDF：ln((N - df + 0.5) / (df + 0.5))
        df = Counter()
        for d in tokenized_docs:
            for w in set(d):
                df[w] += 1
        self.idf = {
            w: np.log((self.N - df[w] + 0.5) / (df[w] + 0.5))
            for w in df
        }

    def score(self, query_tokens, doc_idx):
        """计算某个查询与第 doc_idx 篇文档的 BM25 得分"""
        dl = len(self.docs[doc_idx])            # 当前文档长度
        qf = Counter(query_tokens)              # 查询词频
        total = 0.0
        for q, qtf in qf.items():
            if q not in self.idf:
                continue                        # 词表外忽略
            f = self.freqs[doc_idx].get(q, 0)   # 文档内词频
            if f == 0:
                continue
            K = self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            doc_term = f * (self.k1 + 1) / (f + K)          # 词频饱和项
            qry_term = qtf * (self.k2 + 1) / (qtf + self.k2) # 查询词频项
            total += self.idf[q] * doc_term * qry_term
        return total

    def rank(self, query):
        """给所有文档打分并按分数降序排序"""
        tokens = tokenize(query)
        scores = [self.score(tokens, i) for i in range(self.N)]
        order = sorted(range(self.N), key=lambda i: (-scores[i], i))
        return scores, [i + 1 for i in order]


bm25 = BM25(tokenized_docs, k1=1.5, k2=1.0, b=0.75)
scores, ranked = bm25.rank("cat dog")

print("BM25 查询 'cat dog'")
print("各文档得分:", np.round(scores, 4))
print("排序后的文档编号:", ranked)