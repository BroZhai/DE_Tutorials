
"""
主题：把 LLM 当成"一个函数"——给一段提示词(prompt)，拿回一段答案。
      用它来 清洗数据 / 标注数据 / 造数据 / 画图表 / 分析文本。

本脚本覆盖教程的每一节：
  Part 0  两种获取模型的方式（同一套接口）
  Part 1  调用托管 API
          1.2 第一次调用   1.3 chat/ask 封装 + token 记账
          1.4 zero-shot vs few-shot   1.5 JSON mode
          1.6 temperature 的影响     1.7 流式输出   1.8 成本
  Part 2  自己部署小模型（只讲思路 + 检测代码，不实际下载）
  Part 3  数据预处理：问方法 → 要代码 → 执行代码
  Part 4  数据标注：单条 / 整表 / 算准确率 / 投票 / few-shot
  Part 5  数据增强：合成数据 / self-instruct
  Part 6  数据可视化: LLM(需求, 数据描述) = 画图代码
  Part 7  文本分析：情感 / 分类 / 抽取 / 摘要 / 翻译 / 一次全取
  Part 8  练习

★ 重要说明 ★
教程要连 DeepSeek 的 API。为了让这份脚本**不联网、不用 key 也能跑通**,
我写了一个"假模型" FakeClient，它严格遵守同样的 Chat Completions 接口，
只是答案由简单的规则生成。
想用真模型：在终端执行  export DEEPSEEK_API_KEY=sk-xxxx  再运行本脚本，
脚本会自动切换到真模型，其余所有代码一行都不用改 —— 这正是教程强调的
"同一套接口，换个网址和模型名就行"。

运行： python cs5481_tutorial5_simple.py
"""

import os
import sys
import json
import random

# =====================================================
# Part 0  两种获取模型的方式，一套接口
# =====================================================
#   托管 API(DeepSeek)            自己部署(vLLM 起的小模型)
#   花钱按 token 算                免费（花的是 GPU 时间）
#   效果好                         效果弱但够用(1.5B)
#   数据要离开本机                 数据留在本地
# 两者都说 "Chat Completions" 协议，所以 Python 代码完全一样，
# 只改 base_url 和 model 两个字符串。
#
# 一次请求 = 一串消息 + 几个参数：
#   messages: [{"role":"system","content":"你是..."},      # 长期指令
#              {"role":"user","content":"这次要做什么"},    # 当前任务
#              {"role":"assistant","content":"示例答案"}]   # 可以自己写，即 few-shot (让Agent知道自己的'回答模版')
#   temperature / max_tokens / response_format / stream ...
# 一次响应里我们只用四个东西：
#   response.choices[0].message.content   答案文字
#   response.choices[0].finish_reason     "stop"正常结束 / "length"被截断
#   response.usage.total_tokens           计费用的 token 数
#   response.choices[0].message.role      永远是 "assistant"

# =====================================================
# Part 1.1  准备客户端
# =====================================================
# 教程的做法（真实版）：
#   from openai import OpenAI
#   client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
#   MODEL = "deepseek-chat"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
USE_REAL_MODEL = bool(DEEPSEEK_API_KEY)

if USE_REAL_MODEL:
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    MODEL = "deepseek-chat"
    print(">>> 检测到 API key，使用真实模型:", MODEL)
else:
    client = None          # 下面会替换成假模型
    MODEL = "fake-model-1.5B"
    print(">>> 没有 API key，使用内置的『假模型』离线演示（接口完全一样）")


# -----------------------------------------------------
# 假模型：严格遵守 Chat Completions 的返回结构，答案靠规则生成
# -----------------------------------------------------
class _Msg:
    """模拟 response.choices[0].message"""
    def __init__(self, content):
        self.content = content
        self.role = "assistant"

class _Choice:
    def __init__(self, content, finish_reason, delta=None):
        self.message = _Msg(content)
        self.finish_reason = finish_reason
        self.delta = _Msg(delta) if delta is not None else None

class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c

class _Resp:
    def __init__(self, content, finish_reason, p, c):
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage(p, c)


POSITIVE_WORDS = ["perfect", "early", "amazing", "bright", "great", "lovely",
                  "buy again", "as described", "breeze", "stunning", "loads in under"]
NEGATIVE_WORDS = ["stopped", "late", "crushed", "never", "cheap", "break",
                  "mess", "wooden", "broken", "cracked", "twice", "wrong"]


def _guess_sentiment(text, temperature):
    """假模型的情感判断：数正面词和负面词谁多。
       遇到 'but / however' 这种转折，只看转折后面的部分（英语里重点在后面）。
       temperature 高时故意加随机，用来演示"高温不稳定"。"""
    t = text.lower()
    for mark in [" but ", " however ", " although "]:
        if mark in t:
            t = t.split(mark)[-1]        # 只保留转折之后的内容
            break
    pos = sum(w in t for w in POSITIVE_WORDS)
    neg = sum(w in t for w in NEGATIVE_WORDS)
    if temperature >= 1.0 and (pos + neg) > 0:     # 高温：抽签
        return random.choice(["positive", "negative"])
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


def _fake_answer(messages, temperature, json_mode):
    """根据提示词里的关键词，返回一段假答案"""
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    last_user = [m["content"] for m in messages if m["role"] == "user"][-1]
    low = user_text.lower()
    system_text = " ".join(m["content"] for m in messages if m["role"] == "system").lower()

    # ---- few-shot 分类工单（4.4）：看 system 里定的标签集 + 最后一条用户消息 ----
    if "classify support tickets" in system_text:
        t = last_user.lower()
        if any(k in t for k in ["charg", "invoice", "refund", "paid", "bill"]):
            return "billing"
        if any(k in t for k in ["parcel", "transit", "shipp", "deliver", "arriv"]):
            return "delivery"
        if any(k in t for k in ["broke", "broken", "hinge", "defect", "quality"]):
            return "product quality"
        return "billing"

    # ---- 让它写 matplotlib 代码 ----
    if "matplotlib" in low:
        if "bar" in low:
            return ("import numpy as np\nimport matplotlib.pyplot as plt\n"
                    "regions = ['North', 'South', 'East', 'West']\n"
                    "x = np.arange(len(df['quarter'])); width = 0.2\n"
                    "fig, ax = plt.subplots(figsize=(9, 5))\n"
                    "for i, r in enumerate(regions):\n"
                    "    ax.bar(x + i*width, df[r], width, label=r)\n"
                    "ax.set_title('Quarterly Sales by Region')\n"
                    "ax.set_xlabel('Quarter'); ax.set_ylabel('Sales')\n"
                    "ax.set_xticks(x + width*1.5); ax.set_xticklabels(df['quarter'])\n"
                    "ax.legend(title='Region'); plt.tight_layout(); plt.show()")
        return ("import matplotlib.pyplot as plt\n"
                "plt.plot(df['month'], df['visits'], marker='o', label='Visits')\n"
                "plt.plot(df['month'], df['signups'], marker='s', label='Signups')\n"
                "plt.title('Visits and Signups Over the Months')\n"
                "plt.xlabel('Month'); plt.ylabel('Count')\n"
                "plt.legend(); plt.tight_layout(); plt.show()")

    # ---- 让它写整条清洗函数 ----
    if "def clean(df)" in low or "pandas function clean" in low:
        return ("import pandas as pd\nimport numpy as np\n"
                "def clean(df):\n"
                "    df = df.copy()\n"
                "    df.columns = [c.strip().replace(' ', '_') for c in df.columns]\n"
                "    df = df.drop_duplicates()\n"
                "    df['Created_At'] = pd.to_datetime(df['Created_At'], errors='coerce')\n"
                "    df['unit_price($)'] = pd.to_numeric(df['unit_price($)'], errors='coerce')\n"
                "    df['Region'] = df['Region'].astype(str).str.strip().str.capitalize()\n"
                "    for col in df.select_dtypes(include=[np.number]).columns:\n"
                "        df[col] = df[col].fillna(df[col].median())\n"
                "    df['Created_At'] = df['Created_At'].ffill()\n"
                "    return df")

    # ---- 让它填空缺失值 ----
    if "fill the missing values" in low:
        return "df['units sold'] = df['units sold'].fillna(df['units sold'].median())"

    # ---- 问"有哪些处理方法" ----
    if "techniques" in low and "missing" in low:
        return ("1. 删掉有缺失的行 - df.dropna(subset=['units sold'])\n"
                "2. 填成常数 0     - df['units sold'].fillna(0)\n"
                "3. 填成中位数     - df['units sold'].fillna(df['units sold'].median())")

    if "cleaning pipeline" in low:
        return ("1. 读文件 df = pd.read_csv(path)\n"
                "2. 列名小写、去空格、空格换下划线\n"
                "3. 删掉全空的行和列\n"
                "4. 去重 df.drop_duplicates()\n"
                "5. 文本列去掉首尾空白\n"
                "6. pd.to_numeric(errors='coerce') 把数字文本转成数字\n"
                "7. 缺失值：删掉或者填充\n"
                "8. 重置索引并保存")

    # ---- JSON 模式：按提示词里要的字段返回 JSON ----
    if json_mode:
        if "city" in low and "category" in low:      # 1.5 抽取工单字段
            return json.dumps({"id": "T-91", "city": "Shanghai",
                               "category": "broken payment terminal"})
        if "mapping" in low:
            return json.dumps({"mapping": {
                "order id": "order_id", "Created At": "created_at",
                "Region": "region", "units sold": "units_sold",
                "unit price($)": "unit_price"}})
        if "reviews" in low:
            return json.dumps({"reviews": [
                {"review": "Battery dies within an hour.", "label": "negative"},
                {"review": "Picture quality is stunning.", "label": "positive"},
                {"review": "Earbuds stopped charging.", "label": "negative"},
                {"review": "The keyboard feels amazing.", "label": "positive"},
                {"review": "Heart rate sensor is inaccurate.", "label": "negative"},
                {"review": "Setup was a breeze.", "label": "positive"}]})
        if "tasks" in low:
            return json.dumps({"tasks": [
                {"instruction": "How many minutes are in 3.5 hours?", "output": "210"},
                {"instruction": "What is 15% of 240?", "output": "36"},
                {"instruction": "Convert 100 Fahrenheit to Celsius.", "output": "37.8"},
                {"instruction": "What date is 10 days after 2024-02-20?", "output": "2024-03-01"},
                {"instruction": "What is the area of a 6 by 4 rectangle?", "output": "24"}]})
        if "date" in low and "number" in low and "name" in low:
            return json.dumps({"date": "1 October 2024", "number": 12, "name": "Inception"})
        if "sentiment" in low and "topic" in low:      # 7.6 一次取多个字段
            s = _guess_sentiment(user_text, temperature)
            topic = "other"
            for key, val in [("charg", "billing"), ("refund", "billing"),
                             ("parcel", "delivery"), ("transit", "delivery"),
                             ("deliver", "delivery")]:
                if key in low:
                    topic = val
                    break
            # 从 <ticket>...</ticket> 里把真正的工单正文抠出来做摘要
            body = last_user.split("<ticket>")[-1].split("</ticket>")[0]
            return json.dumps({"sentiment": s, "topic": topic,
                               "entities": [],
                               "summary": " ".join(body.split()[:8]) + " ..."})
        if "topic" in low:
            topic = "other"
            for key, val in [("charg", "billing"), ("refund", "billing"), ("invoice", "billing"),
                             ("parcel", "delivery"), ("transit", "delivery")]:
                if key in low:
                    topic = val
                    break
            return json.dumps({"topic": topic})
        if "label" in low:
            s = _guess_sentiment(user_text, temperature)
            return json.dumps({"label": s, "confidence": 0.95,
                               "reason": "key words in the review"})
        return json.dumps({"answer": "ok"})

    # ---- 普通文字回答 ----
    if "translate" in low and "chinese" in low:
        return "数据质量问题通常是在模型部署之后才被发现的。"
    if "summar" in low:
        return ("市议会通过新交通方案：更换有轨电车、新增 40 公里公交专用道，"
                "并提高市中心停车费来筹资；工程 3 月开工。")
    if "sentiment" in low:
        return _guess_sentiment(user_text, temperature)
    if "name for a dataset" in low:
        return "**CityCycle**"
    if "data quality problems" in low:
        return "Missing values\nDuplicate records\nInconsistent formatting"
    if "hello" in low:
        return "hello from the model"
    return "（假模型）我已收到你的问题：" + user_text[:40] + " ..."


class FakeCompletions:
    def create(self, model, messages, temperature=0.0, max_tokens=1000,
               response_format=None, stream=False):
        json_mode = bool(response_format)
        text = _fake_answer(messages, temperature, json_mode)
        # 高温时给"起名"这类开放问题加点花样，直观展示 temperature 的作用
        if temperature > 1.0 and "name for a dataset" in \
                " ".join(m["content"] for m in messages if m["role"] == "user").lower():
            text = ("**CityCycle** - short and catchy. Alternatives: "
                    "**UrBike** (techy), **VeloCity** (European flavour).")
        prompt_tok = max(1, len(" ".join(m["content"] for m in messages)) // 4)
        comp_tok = max(1, len(text) // 4)

        # 模拟 max_tokens 截断：答案太长就砍掉，并把 finish_reason 设成 "length"
        finish = "stop"
        if comp_tok > max_tokens:
            text = text[: max_tokens * 4]
            comp_tok = max_tokens
            finish = "length"

        if stream:                                   # 流式：一小块一小块吐出来
            def gen():
                step = 12
                for i in range(0, len(text), step):
                    piece = text[i:i + step]
                    r = _Resp(piece, "stop", prompt_tok, 1)
                    r.choices[0].delta = _Msg(piece)
                    yield r
            return gen()
        return _Resp(text, finish, prompt_tok, comp_tok)


class FakeChat:
    completions = FakeCompletions()

class FakeClient:
    chat = FakeChat()


if not USE_REAL_MODEL:
    client = FakeClient()


# =====================================================
# Part 1.3  两个会一直复用的小函数
# =====================================================
# 教程把那长长的一串调用包成 chat() 和 ask()，后面所有小节都用 ask()。
TOKEN_LOG = []          # 每次调用把 token 数记进来，方便最后算账


def chat(messages, temperature=0.0, max_tokens=1000, json_mode=False):
    """发一串消息给模型，返回答案文字。"""
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature,
        max_tokens=max_tokens, **extra)
    TOKEN_LOG.append(response.usage.total_tokens)
    if response.choices[0].finish_reason == "length":
        print("   (警告：答案被截断了，把 max_tokens 调大一点)")
    return response.choices[0].message.content


def ask(prompt, system=None, temperature=0.0, max_tokens=1000, json_mode=False):
    """只问一个问题，拿一个答案。后面全篇都用它。"""
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, temperature=temperature, max_tokens=max_tokens,
                json_mode=json_mode)


def tokens_used():
    """从脚本开始到现在一共用了多少 token。"""
    total = 0
    for n in TOKEN_LOG:
        total += n
    return total


# =====================================================
# Part 1.2  第一次调用：给一条评论做情感分类
# =====================================================
print("\n" + "=" * 66)
print("1.2  第一次调用：把一条评论分成 positive / negative")
print("=" * 66)
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a data annotation assistant. Answer with one word."},
        {"role": "user", "content": "Classify the sentiment of this review as positive or negative.\n"
                                    "Review: 'The plot was a mess and the acting was wooden.'\n"
                                    "Sentiment:"},
    ],
    temperature=0.0,    # 0 = 尽量稳定，做标注时就用 0
    max_tokens=10,      # 只要一个词，给 10 个 token 够了
)
print("答案        :", resp.choices[0].message.content)
print("结束原因    :", resp.choices[0].finish_reason)   # stop=正常，length=被截断
print("本次 token  :", resp.usage.total_tokens)
print("提醒：模型没有记忆，每次调用都要把整个对话重发一遍，所以发送的数据要短。")


# =====================================================
# Part 1.4  提示词工程：zero-shot vs few-shot
# =====================================================
print("\n" + "=" * 66)
print("1.4  提示词的两种写法：直接问(zero-shot) vs 给例子(few-shot)")
print("=" * 66)
# 好提示词四件套：1 角色(system)  2 任务  3 数据(用 <review> 包起来)  4 例子
review = "Battery life is amazing but the screen cracked after two days."

zero_shot = ("Classify the sentiment of the review as positive or negative.\n"
             "<review>" + review + "</review>\n")

few_shot = ("Classify the sentiment of the review as positive or negative.\n\n"
            "<review>Delivery was fast and the packaging was perfect.</review>\npositive\n\n"
            "<review>The item arrived broken and support never replied.</review>\nnegative\n\n"
            "<review>" + review + "</review>\n")
# few-shot 的关键：assistant 那几轮是我们自己写的示范答案，用来"教"它格式

print("zero-shot :", ask(zero_shot, system="Answer with one word.", max_tokens=10))
print("few-shot  :", ask(few_shot, system="Answer with one word.", max_tokens=10))
print("(few-shot 通常更稳，因为例子顺便把答案格式也定下来了)")


# =====================================================
# Part 1.5  JSON 模式：让答案直接变成一行数据
# =====================================================
print("\n" + "=" * 66)
print("1.5  JSON 模式：答案不再是散文，而是可以直接解析成字典的数据")
print("=" * 66)
record = ask(
    "Extract the fields id, city and category from this ticket.\n"
    "Ticket: <t>ticket T-91, filed in Shanghai, about a broken payment terminal</t>\n"
    'Return JSON with the keys "id", "city" and "category".',
    temperature=0.0, max_tokens=200, json_mode=True,   # json_mode 就是 response_format
)
print("原始答案   :", record)
print("转成字典   :", json.loads(record))
print("→ 有了字典就能塞进 DataFrame 变成一行，这就是'LLM 当数据转换器'。")


# =====================================================
# Part 1.6  temperature 的作用
# =====================================================
print("\n" + "=" * 66)
print("1.6  temperature：0 = 每次都挑最可能的词（稳定）；越大越随机（有创意）")
print("=" * 66)
q = "Invent a short name for a dataset about city bicycles."
print("temperature 0.0 :", ask(q, temperature=0.0, max_tokens=8))
print("temperature 1.3 :", ask(q, temperature=1.3, max_tokens=300))
print("经验：抽取/标注/写代码用 0；头脑风暴、造数据用 0.7~1.0。")


# =====================================================
# Part 1.7  流式输出
# =====================================================
print("\n" + "=" * 66)
print("1.7  流式输出 stream=True：答案一小块一小块地回来（就像打字机）")
print("=" * 66)
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "List three data quality problems, one per line."}],
    temperature=0.7, max_tokens=120, stream=True,
)
print("流式拼出来的答案：", end="")
for piece in stream:
    text = piece.choices[0].delta.content      # delta 里只有"新增的那一小段"
    if text is not None:
        print(text, end="")
print()


# =====================================================
# Part 1.8  token 与成本
# =====================================================
print("\n" + "=" * 66)
print("1.8  成本 = 输入的 token + 输出的 token，都按单价计费")
print("=" * 66)
print("到目前为止用掉的 token:", tokens_used())
print("成本公式： (prompt_tokens * 输入单价 + completion_tokens * 输出单价) / 1_000_000")
print("省钱三招：①别整表塞进提示词，只发 describe(df) 这种摘要；")
print("          ②不变的部分放提示词开头（很多厂商对重复前缀有缓存折扣）；")
print("          ③标注成千上万行时别用 reasoner（它会先写一大段思考，也要钱）。")


# =====================================================
# Part 2  自己部署小模型（仅演示判断逻辑）
# =====================================================
print("\n" + "=" * 66)
print("2    自己部署：vLLM 一条命令就能起一个 OpenAI 同款接口的服务")
print("=" * 66)
print("  !pip install vllm")
print("  !vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 &")
print("服务起来后，改两行就能用，其它代码一个字都不用动：")
print("  client = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='not-needed')")
print("  MODEL  = 'Qwen/Qwen2.5-1.5B-Instruct'")
try:
    import requests
    r = requests.get("http://127.0.0.1:8000/v1/models", timeout=2)
    print("检测到本地服务：", r.json())
except Exception:
    print("（本机 8000 端口没有服务，这是正常的：本脚本用假模型继续往下跑）")


# =====================================================
# Part 3  用 LLM 做数据预处理
# =====================================================
import pandas as pd
import numpy as np

print("\n" + "=" * 66)
print("3    数据预处理：一份故意做脏的表")
print("=" * 66)
messy = pd.DataFrame({
    "order id":     ["o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9", "o10", "o11", "o12"],
    "Created At":   ["2024-01-05", "05/01/2024", "2024-01-07", None, "2024-02-01",
                     "2024-02-03", "2024-02-03", "2024-03-11", "2024-03-12",
                     "2024-04-02", "2024-04-02", "2024-04-09"],
    "Region":       ["North", "north", "SOUTH", "South ", "East", "east",
                     "West", "west", "North", "North", "North", "East"],
    "units sold":   [10, 12, 7, 9, 40, np.nan, 15, 22, 18, 999999, 20, np.nan],
    "unit price($)":["9.5", "9.5", "12.0", "12", "7.25", "7.25",
                     "7.25", "5.0", "5", "5.0", "5.0", "5.0"],
})
# 脏在哪：重复行、缺值、数字存成文本、999999 异常值、大小写/空格不一致、日期写成文本
print(messy)


# ---------- 3.1 先问"通用清洗流程" ----------
print("\n--- 3.1 让它先给一份通用清洗清单 ---")
print(ask("Give a generic pandas data cleaning pipeline for a CSV file that may contain "
          "missing values, duplicated rows, inconsistent column names, and numbers stored "
          "as text. Answer as a numbered list of at most 8 short lines.",
          system="You are a concise data engineering assistant.", max_tokens=400))


# ---------- 3.2 先问方法，再要代码，然后执行 ----------
print("\n--- 3.2 分两步：先问有哪些方法，再为选中的方法要代码 ---")
print(ask("I have a dataframe with a numeric column called 'units sold' that has missing "
          "values. List 3 techniques for handling those missing values, one per line, "
          "and say which pandas function implements each one.", max_tokens=300))


def extract_code(text):
    """模型爱写解释，我们只留下 ``` 围栏里的第一段代码。"""
    if "```" not in text:
        return text.strip()
    code = text.split("```")[1]
    if code.startswith("python"):
        code = code[6:]
    return code.strip()


def run_code(code, df):
    """执行模型写的代码。约定：代码处理完后把结果留在变量 df 里。"""
    space = {"df": df, "pd": pd, "np": np}     # 只允许它看见这几个变量
    exec(extract_code(code), space)
    return space["df"]


code = ask("Fill the missing values of the column 'units sold' with the median.\n"
           "The dataframe is already in a variable called df.\n"
           "Return only python code, no explanation.",
           temperature=0.0, max_tokens=200)
print("模型给的代码:", extract_code(code))
cleaned = run_code(code, messy.copy())
print("执行后还剩几个缺失值:", cleaned["units sold"].isna().sum())
print("注意：模型写的代码权限和你一样，先读一遍再跑；第一次跑挂了很正常，")
print("      把报错信息发回去让它改，这是标准操作。")


# ---------- 3.3 让它写整条流水线 ----------
print("\n--- 3.3 描述整张表，让它一次写出 clean(df) 函数 ---")


def describe(df, n=3):
    """给 DataFrame 写一份又短又便宜的说明书，放进提示词里。
       千万别把整张表塞进去——那是烧钱最快的方式。"""
    text = "shape: " + str(df.shape) + "\n"
    text += "columns and types:\n" + df.dtypes.to_string() + "\n"
    text += "missing values:\n" + df.isna().sum().to_string() + "\n"
    text += "first rows:\n" + df.head(n).to_string()
    return text


instructions = ("Write a pandas function clean(df) for the dataframe described below. It should\n"
                "- strip the column names and replace spaces with underscores\n"
                "- drop duplicated rows\n"
                "- parse the column 'Created At' to datetime\n"
                "- convert the column 'unit price($)' to numbers\n"
                "- strip whitespace and fix the capitalisation of the column 'Region'\n"
                "- fill missing numbers with the median and missing dates with the previous value\n"
                "Return only python code: the imports and the function clean(df).\n\n"
                "Dataframe:\n" + describe(messy))
code = ask(instructions, system="You are a senior data engineer. Output runnable pandas code only.",
           temperature=0.0, max_tokens=900)
print(extract_code(code))

try:
    namespace = {"pd": pd, "np": np}
    exec(extract_code(code), namespace)
    cleaned = namespace["clean"](messy.copy())     # 要的是函数，那就调用它
    print("\n执行结果（前 4 行）：")
    print(cleaned.head(4))
    print("\n★ 一定要自己看结果：教程里模型用了 dayfirst=True，把 2024-01-05")
    print("  解析成了 5 月 1 日；999999 那个异常值也没被处理掉。")
    print("  看起来对的代码，不一定真的对——跑一遍、看几行、再修。")
except Exception as error:
    print("模型给的代码跑挂了：", error)


# ---------- 3.4 ask_json：答案直接变成字典 ----------
print("\n--- 3.4 ask_json：要 JSON，并负责解析；失败就再要一次 ---")


def ask_json(prompt, system=None, temperature=0.0):
    """向模型要一个 JSON 对象，返回 Python 字典。"""
    for _ in range(3):
        text = ask(prompt, system=system, temperature=temperature, json_mode=True)
        try:
            return json.loads(extract_code(text))
        except Exception:
            prompt = prompt + "\n\nYour last answer was not valid JSON. Return JSON only."
    print("(模型始终没给出合法 JSON)")
    return {}


raw = ask_json("Map each messy column name below to a standard snake_case name.\n"
               "Columns: " + str(list(messy.columns)) + "\n"
               'Return JSON of the form {"mapping": {"<messy name>": "<standard name>"}}')
print("列名映射:", raw)
mapping = raw.get("mapping", {})
print("改名后  :", list(messy.rename(columns=mapping).columns))


# =====================================================
# Part 4  用 LLM 做数据标注
# =====================================================
print("\n" + "=" * 66)
print("4    数据标注：这是 LLM 最省人力的一环")
print("=" * 66)
labelled = [
    ["The packaging was perfect and it arrived a day early.", "positive"],
    ["It stopped charging after one week.", "negative"],
    ["Exactly as described, would buy again.", "positive"],
    ["Delivery was late and the box was crushed.", "negative"],
    ["The battery life is amazing, easily two days.", "positive"],
    ["Support never answered my emails.", "negative"],
    ["Cheap plastic, feels like it will break soon.", "negative"],
    ["The screen is bright and the colours are great.", "positive"],
]
reviews = pd.DataFrame(labelled, columns=["review", "gold"])   # gold = 我人工标的答案


def annotate(text):
    """标一条：要标签 + 置信度 + 一句话理由。"""
    prompt = ("Label the sentiment of the review below.\n"
              "<review>" + text + "</review>\n"
              "Allowed labels: positive, negative.\n"
              'Return JSON: {"label": "positive" or "negative", '
              '"confidence": a number from 0 to 1, "reason": "at most 12 words"}')
    return ask_json(prompt, system="You are a careful data annotator.")


print("--- 4.1 标一条 ---")
print(annotate("The plot was a mess and the acting was wooden."))

print("\n--- 4.2 标整张表，并和人工标签对比算准确率 ---")
labels, confs, reasons = [], [], []
for text in reviews["review"]:
    r = annotate(text)
    labels.append(r.get("label"))
    confs.append(r.get("confidence"))
    reasons.append(r.get("reason"))
reviews["pred"] = labels
reviews["confidence"] = confs
reviews["reason"] = reasons
# 两列比较得到 True/False，.mean() 就是把 True 当 1 求平均 → 准确率
print("在我这 8 条人工标签上的准确率:", round((reviews["pred"] == reviews["gold"]).mean(), 2))
print(reviews[["review", "gold", "pred", "confidence"]].to_string())
print("提示：置信度低的、或和你不一致的行，就是该人工复查的行。")

print("\n--- 4.3 自一致性：同一条问 k 次，取多数票 ---")


def annotate_vote(text, k=3):
    """问 k 次，返回出现最多的标签和票数分布。"""
    votes = [annotate(text).get("label") for _ in range(k)]
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=counts.get)
    return best, counts


tricky = "Battery life is amazing but the screen cracked after two days."
print("混合情感的评论:", tricky)
print("投票结果:", annotate_vote(tricky, k=3))
print("多花几倍 token 换更可靠的结果；票数分散(比如 3:2)说明这行本身就难判。")

print("\n--- 4.4 few-shot：用自己的标签集教它（assistant 消息由我们来写） ---")
our_labels = ["billing", "delivery", "product quality"]
messages = [
    {"role": "system", "content": "Classify support tickets into one of " + str(our_labels) +
                                  ". Answer with the label only."},
    {"role": "user", "content": "I was charged twice for the same order."},
    {"role": "assistant", "content": "billing"},          # ← 我们自己写的示范
    {"role": "user", "content": "The parcel has been 'in transit' for two weeks."},
    {"role": "assistant", "content": "delivery"},
    {"role": "user", "content": "The hinge broke the first time I opened it."},
    {"role": "assistant", "content": "product quality"},
    {"role": "user", "content": "My invoice shows an amount I never agreed to."},
]
print("新工单的分类:", chat(messages, max_tokens=15))


# =====================================================
# Part 5  用 LLM 造数据
# =====================================================
print("\n" + "=" * 66)
print("5    数据增强：好数据稀缺、人工标注贵，那就让模型造")
print("=" * 66)
print("--- 5.1 合成数据（temperature 调高才多样化） ---")
data = ask_json("Generate 6 short customer reviews about electronics: 3 negative and 3 positive. "
                "Vary the products and the reasons. "
                'Return JSON: {"reviews": [{"review": "...", "label": "positive" or "negative"}]}',
                temperature=0.9)
synthetic = pd.DataFrame(data.get("reviews", []), columns=["review", "label"])
print(synthetic.to_string())

hand_rows = reviews[["review", "gold"]].rename(columns={"gold": "label"})
augmented = pd.concat([hand_rows, synthetic], ignore_index=True)   # 拼起来就完事
print("\n合并后的类别分布:")
print(augmented["label"].value_counts().to_string())
print("总行数:", len(augmented))

print("\n--- 5.2 self-instruct：给它几个种子任务，让它自己发明新任务 ---")
seeds = [{"instruction": "Convert 25 degrees Celsius to Fahrenheit.", "output": "77"},
         {"instruction": "What date is three days after 2024-02-27?", "output": "2024-03-01"}]
d = ask_json("Here are two example tasks:\n" + json.dumps(seeds, indent=1) +
             "\n\nInvent 5 new tasks in exactly the same style. Do not repeat the examples. "
             'Return JSON: {"tasks": [{"instruction": "...", "output": "..."}]}',
             temperature=1.0)
print(pd.DataFrame(d.get("tasks", []), columns=["instruction", "output"]).to_string())


# =====================================================
# Part 6  用 LLM 画图
# =====================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)   # 屏蔽 plt.show() 在无界面下的提示

print("\n" + "=" * 66)
print("6    数据可视化：画图代码 = LLM(需求, 数据描述)")
print("=" * 66)
sales = pd.DataFrame({"quarter": ["Q1", "Q2", "Q3", "Q4"],
                      "North": [120, 135, 150, 172], "South": [98, 104, 99, 130],
                      "East": [143, 150, 168, 181], "West": [77, 88, 95, 112]})
metrics = pd.DataFrame({"month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                        "visits": [1200, 1500, 1700, 1600, 2100, 2400],
                        "signups": [60, 75, 88, 80, 104, 122],
                        "revenue": [4200, 5100, 6000, 5700, 7400, 8600]})
os.makedirs("output_plots", exist_ok=True)


def llm_plot(requirement, df, filename):
    """说清楚想要什么图 + 把数据表描述一下 → 模型给代码 → 我们执行。"""
    prompt = ("Write matplotlib code that draws the chart described below.\n"
              "The requirement: " + requirement + "\n\n"
              "The dataframe is already in a variable called df:\n" + describe(df, n=len(df)) + "\n\n"
              "Rules: use matplotlib only (it is already imported), use the dataframe df, "
              "add a title and axis labels, and finish with plt.show(). Return only python code.")
    code = extract_code(ask(prompt, system="You are a data visualization expert.",
                            max_tokens=900))
    print("--- 模型写的代码 ---")
    print(code)
    try:
        exec(code, {"df": df, "pd": pd, "np": np, "plt": plt})
        plt.savefig(os.path.join("output_plots", filename), dpi=80, bbox_inches="tight")
        plt.close()
        print("已保存图片 ->", filename)
    except Exception as error:
        print("模型给的代码跑挂了：", error)
    print("注意：图好看不等于分析对，模型可能画出误导性的坐标轴；")
    print("      要结论就再单独问一次。")


llm_plot("a grouped bar chart: one group of bars per quarter, one bar per region",
         sales, "01_grouped_bar.png")
llm_plot("a line chart of visits and signups over the months, with markers",
         metrics, "02_line.png")


# =====================================================
# Part 7  用 LLM 做文本分析
# =====================================================
print("\n" + "=" * 66)
print("7    文本分析：情感 / 分类 / 抽取 / 摘要 / 翻译，全都是提示词")
print("=" * 66)
tickets = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "text": ["I was charged twice for the same order, please refund the second charge.",
             "The parcel has been 'in transit' for two weeks, where is it?",
             "The next public holiday is on 1 October 2024, and there are 12 new exhibits.",
             "Great news: the app finally loads in under a second after the update!",
             "Inception was directed by Christopher Nolan."],
})

print("--- 7.1 情感分析：一句话就够 ---")
print(ask("What is the sentiment of this text: 'The battery lasts two days and the screen "
          "is lovely.' Answer with one word: positive, negative or neutral.", max_tokens=10))

print("\n--- 7.2 文本分类：标签集一定要自己定死，否则答案没法统计 ---")
topics = ["billing", "delivery", "product information", "other"]


def classify(text):
    prompt = ("Classify the text below into exactly one of these topics: " + str(topics) + ".\n"
              "<text>" + text + "</text>\n"
              'Return JSON: {"topic": "<one of the topics>"}')
    return ask_json(prompt, system="You are a text classification engine.").get("topic", "other")


tickets["topic"] = [classify(t) for t in tickets["text"]]
print(tickets[["id", "topic"]].to_string())

print("\n--- 7.3 信息抽取：一个 schema 一次抽好几个字段 ---")
print(ask_json("Extract information from the sentence below.\n"
               "<sentence>The next public holiday is on 1 October 2024, there are 12 new "
               "exhibits, and the film Inception was directed by Christopher Nolan.</sentence>\n"
               'Return JSON with exactly these keys: "date", "number", "name".'))

print("\n--- 7.4 摘要 ---")
long_text = ("The city council approved the new transit plan on Tuesday after two years of "
             "debate. The plan replaces the ageing tram fleet, adds 40 km of dedicated bus "
             "lanes and raises parking fees in the city centre to pay for it. Supporters say "
             "it will cut commuting times by a fifth; critics say the parking fees will hurt "
             "small shops. Construction is scheduled to start in March.")
print(ask("Summarise the text in at most two sentences.\n<text>" + long_text + "</text>",
          max_tokens=200))

print("\n--- 7.5 翻译 ---")
print(ask("Translate the sentence below into Chinese. Output only the translation.\n"
          "<text>Data quality problems are usually found after the model is deployed.</text>",
          max_tokens=100))

print("\n--- 7.6 一次调用取回所有字段，再循环整张表（最推荐的做法） ---")


def analyze(text):
    prompt = ("Analyse the support ticket below and return one JSON object with the keys:\n"
              ' "sentiment": one of ["positive", "negative", "neutral"],\n'
              ' "topic": one of ["billing", "delivery", "product information", "other"],\n'
              ' "entities": a list of the important names, dates or amounts (may be empty),\n'
              ' "summary": a summary of at most 10 words.\n'
              "<ticket>" + text + "</ticket>")
    return ask_json(prompt, system="You are a text analysis engine that always answers with JSON.")


records = [analyze(t) for t in tickets["text"]]
tickets["sentiment"] = [r.get("sentiment") for r in records]
tickets["summary"] = [r.get("summary") for r in records]
print(tickets[["id", "topic", "sentiment", "summary"]].to_string())
print("\n到此它又是一张普通的表了，可以照常做统计：")
print(tickets.groupby(["topic", "sentiment"]).size().to_string())
print("\n工程提醒：①提示词要短；②把答案缓存下来，只重算变过的行；")
print("          ③凡是你要拿来统计的，都用 temperature=0；④把原始答案单独存一列。")


# =====================================================
# Part 8  练习
# =====================================================
print("\n" + "=" * 66)
print("8    练习：temperature 与正确性")
print("=" * 66)
mixed = "Battery life is amazing but the screen cracked after two days."
for temp, times in [(0.0, 10), (1.2, 10)]:
    votes = [annotate(mixed).get("label") if False else
             ask("Classify the sentiment: <review>" + mixed + "</review> Answer with one word.",
                 temperature=temp, max_tokens=10) for _ in range(times)]
    print(f"temperature={temp}  问 {times} 次 →", pd.Series(votes).value_counts().to_dict())
print("结论：要标一万条评论，用 temperature=0 —— 稳定、可复现、可统计。")

print("\n" + "=" * 66)
print("全部演示结束。全程共用 token:", tokens_used())
print("=" * 66)
