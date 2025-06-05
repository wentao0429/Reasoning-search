# -*- coding: utf-8 -*-
"""Stage‑2 ─ QA Generation & Filtering
======================================
Input  :  ./data/<bundle_id>.json              ← output of crawler_pipeline.py
Output :  ./qa_dataset.jsonl  (≈ 5‑7e2 lines)  ← one JSON per QA pair

Pipeline ---------------------------------------------------------------
1.  Walk all bundle files (JSON, list of docs per bundle)
2.  Build prompt from ≤K docs / bundle; call OpenAI ChatCompletion (gpt‑4o)
3.  Receive JSON list [{question,answer,type,support}]  in *简体/英文混排*
4.  Normalize → 简体；trim answer length
5.  Novelty filter: call LLM 无上下文回答 → embed cosine(sim) < τ (0.75)
6.  Complexity filter: len(support) ≥ 2  OR  keep_rate_single ~ 0.15
7.  Deduplicate by exact question string
8.  Append to qa_dataset.jsonl until 5‑7.5e2 samples

Requirements -----------------------------------------------------------
$ export OPENAI_API_KEY=...
> pip install openai tiktoken sentence_transformers opencc-python-reimplemented tqdm
"""
from __future__ import annotations

import json, os, re, random, glob, time, logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import openai
import tiktoken
from sentence_transformers import SentenceTransformer, util
from opencc import OpenCC
from tqdm import tqdm

# ---------------------------------------------------------------------
# 0. CONFIG ------------------------------------------------------------
# ---------------------------------------------------------------------
CLUSTER_FILE = Path("news_clusters_full.json")  # 聚类文件路径
OUT_FILE   = Path("qa_dataset_full.jsonl")
MODEL_GEN  = "gpt-4.1-mini"              # generation model (调整 quota)
MODEL_EVAL = "gpt-4o-mini"       # novelty check (cheap)
EMBED_MODEL= "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DOCS_PER_PROMPT = 4                     # ≤4 docs per bundle to control tokens
MAX_QUESTIONS_PER_PROMPT = 3            # let LLM output 3 Q/A
TARGET_MIN, TARGET_MAX = 500, 800
NOVELTY_THRESHOLD = 0.75               # cosine similarity upper bound
SINGLE_SUPPORT_RATE = 0.15             # allow up to 15 % single‑doc questions

# 配置 OpenAI API
client = openai.OpenAI(
    api_key="",
    base_url="https://api.openai-proxy.com/v1"
)
assert client.api_key, "请先设置 OPENAI_API_KEY 环境变量"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------
# 1. INIT MODELS -------------------------------------------------------
# ---------------------------------------------------------------------
emb_model = SentenceTransformer(EMBED_MODEL)
opencc_ts = OpenCC("t2s")               # 繁→简
enc = tiktoken.encoding_for_model("gpt-4o-mini")

# ---------------------------------------------------------------------
# 2. HELPERS -----------------------------------------------------------
# ---------------------------------------------------------------------
PROMPT_TEMPLATE_SYS = """你是一名资深出题专家，要为【多跳检索训练集】批量生成高质量中文 QA。

===========================
一、评分标准（Rubric）
===========================
R1. **信息完整**：问题只能在综合全部给定资料后才能回答。
R2. **题型限定**（使用最适合这些资料的题型）
    • fact  : 单一数字 / 专有名词，答案 ≤60 字
    • cause : 2–3 句因果或影响链，答案 ≤100 字
    • multi : 多选 / 列表，答案 ≤120 字，必须包含完整选项列表，若为多选题需标出正确选项
R3. **答案要求**：
    – 全部简体中文，不出现口水词（如"综上所述""可以看到"）
    – 精炼、信息密度高，不添加无资料依据的推测
    – 答案必须严格基于给定资料，不能包含常识性推测
R4. **support 字段**：列出支撑答案所需的 doc_id，至少 2 篇
R5. **格式**：仅输出 JSON 数组，每题包含 {question, answer, type, support}

===========================
二、示例
===========================
【fact 示例】
{
  "question": "2025 年 2 月 20 日，美联储将联邦基金利率下调多少个基点？",
  "answer": "下调 50 个基点，目标区间降至 4.75%–5.00%。",
  "type": "fact",
  "support": ["gnews_20250428_626", "gnews_20250428_674"]
}

【cause 示例】
{
  "question": "特朗普政府在美日韩联合声明中支持台湾参与国际组织将带来哪些连锁影响？",
  "answer": "该立场强化对华强硬路线 → 台湾国际参与空间扩大；同时加剧中韩外交摩擦，韩国需在美中之间更谨慎平衡。",
  "type": "cause",
  "support": ["gnews_20250428_676", "gnews_20250428_693"]
}

【multi 示例】
{
  "question": "以下哪些因素被美日韩三国外长视为提升经济韧性的关键？（多选）\nA. 人工智能合作\nB. 传统制造业升级\nC. 量子技术研发\nD. 金融监管改革\nE. 能源转型",
  "answer": "正确选项：A、C。\nA 人工智能合作；C 量子技术研发。",
  "type": "multi",
  "support": ["gnews_20250428_626", "gnews_20250428_674", "gnews_20250428_676"]
}

===========================
三、注意事项
===========================
1. 问题必须独立完整，不能包含"根据以下文档"等对原文的引用
2. 每个问题必须至少需要两篇文档才能回答
3. 答案必须精炼，去除多余空格和口水词
4. 多选题必须包含完整的选项列表，并明确标注正确选项
5. 问题必须明确具体，不能过于宽泛或需要常识性推测
6. 答案必须严格基于给定资料，不能包含无依据的推测
7. 如果资料不足以支持多源答案，宁可生成单源问题
8. 只输出最终通过的题目数组，格式必须严格符合 JSON 规范"""

PROMPT_TEMPLATE_USER = """以下是同一主题的若干资料，请生成 {n_q} 道题：
```
{context}
```"""

RE_SPACES = re.compile(r"\s+")

def trim_answer(ans: str, limit: int) -> str:
    ans = RE_SPACES.sub(" ", ans).strip()
    # 粗略按汉字长度限制
    while len(ans) > limit:
        ans = ans[: -1]
    return ans

# ---------------------------------------------------------------------
# 3. NOVELTY CHECK -----------------------------------------------------
# ---------------------------------------------------------------------

def is_novel(question: str, reference_answer: str) -> bool:
    # ask eval model 无上下文回答
    try:
        resp = client.chat.completions.create(
            model=MODEL_EVAL,
            temperature=0.0,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        guess = resp.choices[0].message.content.strip()
    except Exception as e:
        logging.warning(f"[EVAL FAIL] {e}")
        return True  # conservative keep
    # embed → similarity
    vec_ref = emb_model.encode(reference_answer, normalize_embeddings=True)
    vec_guess= emb_model.encode(guess,           normalize_embeddings=True)
    sim = float(util.cos_sim(vec_ref, vec_guess))
    return sim < NOVELTY_THRESHOLD

# ---------------------------------------------------------------------
# 4. MAIN LOOP ---------------------------------------------------------
# ---------------------------------------------------------------------

def main():
    qa_count = 0
    single_support_kept = 0
    
    # 读取聚类文件
    with open(CLUSTER_FILE, 'r', encoding='utf-8') as f:
        clusters = json.load(f)
    
    # 将聚类转换为列表并打乱顺序
    cluster_items = list(clusters.items())
    random.shuffle(cluster_items)

    # 创建总进度条，显示目标数量
    pbar = tqdm(total=TARGET_MAX, desc="生成 QA 对")
    
    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for cluster_id, docs in tqdm(cluster_items, desc="处理聚类"):
            if qa_count >= TARGET_MAX:
                break
            if not docs:
                continue
                
            # 尝试生成多源问题
            docs_sorted = sorted(docs, key=lambda d: len(d["content"]), reverse=True)[:DOCS_PER_PROMPT]
            context = "\n---\n".join(
                f"<doc_id={d['id']}>\n{d['title']}\n{d['content'][:1500]}"  # clip long body
                for d in docs_sorted)

            prompt = PROMPT_TEMPLATE_USER.format(n_q=MAX_QUESTIONS_PER_PROMPT, context=context)
            try:
                resp = client.chat.completions.create(
                    model=MODEL_GEN,
                    temperature=0.7,
                    messages=[
                        {"role": "system", "content": PROMPT_TEMPLATE_SYS},
                        {"role": "user",   "content": prompt}
                    ]
                )
            except Exception as e:
                logging.warning(f"[OPENAI] {e}")
                time.sleep(0.1)  # 减少等待时间
                continue
                
            content = resp.choices[0].message.content
            try:
                qa_list = json.loads(content)
            except Exception:
                logging.warning(f"[PARSE FAIL] {cluster_id}")
                continue

            for qa in qa_list:
                if qa_count >= TARGET_MAX:
                    break
                q  = opencc_ts.convert(qa["question"]).strip()
                a  = opencc_ts.convert(qa["answer"]).strip()
                typ= qa.get("type", "fact")
                sup= qa.get("support", [])

                # trim answer by type
                lim = 60 if typ == "fact" else 100 if typ == "cause" else 120
                a = trim_answer(a, lim)

                # 简单 dedup
                if os.path.exists(OUT_FILE):
                    if any(prev_q == q for prev_q in open_prev_questions(OUT_FILE)):
                        continue

                # novelty check
                if not is_novel(q, a):
                    continue
                    
                # 检查多选题格式
                if typ == "multi" and not any(c in q for c in ["A.", "B.", "C.", "D.", "E."]):
                    continue
                    
                # complexity rule
                if len(sup) < 2:
                    if (single_support_kept / max(1, qa_count)) > SINGLE_SUPPORT_RATE:
                        continue
                    single_support_kept += 1
                    
                # dump
                fout.write(json.dumps({"question": q, "answer": a, "type": typ, "support": sup}, ensure_ascii=False)+"\n")
                qa_count += 1
                pbar.update(1)
                
            logging.info(f"cluster {cluster_id} -> keep {qa_count}")
            time.sleep(0.1)
        pbar.close()
    logging.info(f"DONE. total QA: {qa_count}")


def open_prev_questions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["question"]


if __name__ == "__main__":
    main()
