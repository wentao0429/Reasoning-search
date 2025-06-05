import json, glob, os, re
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from umap.umap_ import UMAP
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

### 1. 读入并简单清洗 ----------------------------------------------------
def load_news(folder):
    rows = []
    print(f"正在从文件夹 {folder} 加载数据...")
    for fp in tqdm(glob.glob(os.path.join(folder, '*.json')), desc="处理文件"):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 处理JSON数组
                if isinstance(data, list):
                    print(f"检测到JSON数组，包含 {len(data)} 条记录")
                    for item in data:
                        if not isinstance(item, dict):
                            print(f"警告：数组中的元素不是字典类型")
                            continue
                            
                        title = item.get("标题", "")
                        content = item.get("正文", "")
                        
                        if not title and not content:
                            print(f"警告：记录没有标题和正文内容")
                            continue
                            
                        rows.append({
                            "id": f"{os.path.basename(fp).split('.')[0]}_{len(rows)}",
                            "title": title,
                            "content": content[:1500]  # 过长截断
                        })
                else:
                    print(f"警告：文件 {fp} 的内容不是JSON数组")
                    continue
                    
        except Exception as e:
            print(f"处理文件 {fp} 时发生错误: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            continue
            
    if not rows:
        print("\n详细错误信息：")
        print("1. 检查文件是否存在")
        print("2. 检查文件权限")
        print("3. 检查文件编码")
        print("4. 检查JSON格式")
        raise ValueError("没有成功加载任何有效数据")
        
    df = pd.DataFrame(rows)
    print(f"\n成功加载 {len(df)} 条数据")
    return df

df = load_news('/home/netzone22/data/rlsf')

# 去重（按标题）
df = df.drop_duplicates('title').reset_index(drop=True)

### 2. 生成句向量 ---------------------------------------------------------
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
texts = (df['title'] + '。' + df['content'].str[:512]).tolist()

emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

### 3. 降维 (可选) ---------------------------------------------------------
reducer = UMAP(n_components=50, metric='cosine', random_state=42)
emb_50d = reducer.fit_transform(emb)

### 4. HDBSCAN 初聚 -------------------------------------------------------
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,
    min_samples=2,
    metric='euclidean',
    prediction_data=True
).fit(emb_50d)

df['cluster'] = clusterer.labels_      # -1 = 噪声
print(df['cluster'].value_counts().head())

### 5. 调整簇大小到 2-4 篇 -----------------------------------------------
def split_big_clusters(df, emb, max_size=4):
    new_labels = df['cluster'].copy()
    next_id = new_labels.max() + 1
    for cid, sub in df.groupby('cluster'):
        if cid == -1:        # 先跳过噪声
            continue
        if len(sub) > max_size:
            k = len(sub) // max_size + (len(sub) % max_size > 0)
            km = KMeans(n_clusters=k, random_state=42).fit(emb[sub.index])
            for idx, lbl in zip(sub.index, km.labels_):
                new_labels.at[idx] = next_id + lbl
            next_id += k
    return new_labels

df['cluster'] = split_big_clusters(df, emb_50d)

### 6. 合并孤立点（可选） -----------------------------------------------
def merge_singletons(df, emb, max_size=4):
    # 找到剩下的 -1 或 size==1
    singleton_idx = df[(df['cluster'] == -1) | (df.groupby('cluster')['cluster'].transform('size') == 1)].index
    for idx in singleton_idx:
        vec = emb[idx].reshape(1, -1)
        # 找最近的簇
        clusters = df[~df['cluster'].isin([-1])]['cluster'].unique()
        sims = []
        for cid in clusters:
            members = df[df['cluster'] == cid].index
            if len(members) >= max_size:
                sims.append(-1)        # 已满不考虑
                continue
            # 取簇中心
            center = emb[members].mean(axis=0, keepdims=True)
            sims.append(cosine_similarity(vec, center)[0, 0])
        if sims and max(sims) > 0.55:
            best = clusters[int(np.argmax(sims))]
            df.at[idx, 'cluster'] = best
    return df

df = merge_singletons(df, emb_50d)

### 7. 输出 ---------------------------------------------------------------
# 按簇大小排序
cluster_sizes = df['cluster'].value_counts()
print("\n聚类结果统计：")
print(f"总文章数：{len(df)}")
print(f"噪声文章数：{len(df[df['cluster'] == -1])}")
print(f"有效簇数：{len(cluster_sizes[cluster_sizes > 1])}")
print("\n最大的5个簇：")
print(cluster_sizes.head())

# 为每个簇添加主题信息
def get_cluster_theme(cluster_df):
    # 取标题中最常见的词作为主题
    titles = ' '.join(cluster_df['title'].tolist())
    words = titles.split()
    if not words:
        return "未知主题"
    from collections import Counter
    common_words = Counter(words).most_common(3)
    return ' '.join([word for word, _ in common_words])

# 输出每个簇的详细信息
print("\n簇详细信息：")
for cluster_id, group in df.groupby('cluster'):
    if cluster_id == -1 or len(group) < 2:
        continue
    print(f"\n簇 {cluster_id} (包含 {len(group)} 篇文章):")
    print(f"主题: {get_cluster_theme(group)}")
    print("文章标题示例:")
    for idx, row in group.head(3).iterrows():
        print(f"- {row['title']}")

# 保存完整的聚类结果
def save_cluster_results(df):
    results = {}
    for cluster_id, group in df.groupby('cluster'):
        if cluster_id == -1 or len(group) < 2:
            continue
        results[str(cluster_id)] = group[['id', 'title', 'content']].to_dict('records')
    
    with open('news_clusters_full.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存完整聚类结果到 news_clusters_full.json")

# 保存聚类结果（包含完整文章信息）
save_cluster_results(df)

# 同时保存ID列表版本（用于兼容性）
cluster_groups = df.groupby('cluster')['id'].apply(list)
cluster_groups = cluster_groups[cluster_groups.apply(lambda x: 2 <= len(x) <= 4)]
cluster_groups.to_json('news_clusters.json', force_ascii=False, indent=2)

