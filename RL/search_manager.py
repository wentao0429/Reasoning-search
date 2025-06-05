import re
import logging
import json
import os 
import time 
import hashlib 
import pickle 
from datetime import datetime, timedelta 
from typing import List, Dict, Any, Tuple, TYPE_CHECKING, Optional, Union 
from collections import deque, defaultdict
import networkx as nx  
from accelerate import Accelerator 
import torch 
import random  

# Global configuration parameters
MAX_SNIPPET = 300   # Increase summary length to provide more context
SEARCH_TIMEOUT = 15  # Search timeout in seconds
MAX_RETRIES = 1     # Maximum retry count
RETRY_WAIT_MIN = 1  # Minimum retry wait time (seconds)
RETRY_WAIT_MAX = 5  # Maximum retry wait time (seconds)
SEARCH_TOPK = 3     # Number of search results to return
USE_GENERAL_FOR_ACADEMY = False  # Whether to use general search engine instead of ArXiv for academic content

# Ablation test config: whether to force all searches to use general (web search)
USE_GENERAL_FOR_ALL = False  # When True, news and academic searches will also use general search

# Time information global configuration
ADD_TIME_TO_SEARCH = False  # Whether to add time information to search queries
TIME_SUFFIX_ZH = "2024年10月"  # Time suffix for Chinese queries
TIME_SUFFIX_EN = "Oct 2024"  # Time suffix for English queries

# Add more granular date range control
USE_DATE_RANGE = False  # Whether to use date range control
DATE_RANGE_START = "2024-10-01"  # Date range start (format: YYYY-MM-DD)
DATE_RANGE_END = "2024-11-01"  # Date range end (format: YYYY-MM-DD)
DATE_RANGE_FOR_TYPES = ["general"]  # Search types to apply date range to, options: "general", "news", "academy"

# Full webpage content retrieval configuration
FETCH_FULL_CONTENT = False  # Whether to fetch full webpage content for search results
FETCH_CONTENT_TOPK = 2      # Number of top search results to fetch full content for
FETCH_TIMEOUT = 8           # Timeout for fetching full content (seconds)

# Cache configuration parameters
CACHE_ENABLED = True  # Whether to enable caching
CACHE_DIR = "search_cache_train"  # Cache directory
CACHE_EXPIRY = 30  # Cache expiry in days, None means never expire

# New: Control whether to enable first-stage output format fixing logic
ENABLE_FORMAT_FIXING = True # Default is True to maintain existing behavior

# DataImpulse proxy configuration
# DataImpulse gateway
PROXY_HOST = "gw.dataimpulse.com"
PROXY_PORT = 823
LOGIN = "YOUR_LOGIN"  # Replace with your login credentials
PASSWORD = "YOUR_PASSWORD"  # Replace with your password

PROXY_URL = f"http://{LOGIN}:{PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"
# Add proxy dictionary, compatible with requests and duckduckgo-search
PROXY_DICT = {
    "http": PROXY_URL,
    "https": PROXY_URL,
}
# Proxy string format for async requests
AIO_PROXY = PROXY_URL

# Optional: Set environment variables for global effect (uncomment to enable)
# import os
# os.environ["HTTP_PROXY"] = PROXY_URL
# os.environ["HTTPS_PROXY"] = PROXY_URL
# Log environment variable settings
logger = logging.getLogger(__name__)


import asyncio
import aiohttp
from tenacity import AsyncRetrying, stop_after_attempt, wait_random_exponential, retry, wait_exponential, stop_after_delay, TryAgain, RetryError

# New: Handle forward reference type hints
if TYPE_CHECKING:
    from train_trl import SearchAwareGRPOTrainer
    
def _truncate(txt, n=MAX_SNIPPET):
    return txt if len(txt) <= n else txt[:n] + "..."


# New regular expressions
_SPACE_RE = re.compile(r'[ \t\r]+')
_PUNCT_RE = re.compile(r'[，；、]')        # Convert Chinese commas/semicolons to ;
_EOL_RE   = re.compile(r'\n+')

def canonicalize_search_block(block: str) -> str:
    """
    Normalize <search> internal text for easier regex processing.
    - All tabs/consecutive spaces → single space
    - Chinese punctuation → English
    - Consecutive empty lines / extra spaces → single \n
    - Add semicolons at line end for multiple edges / missing ;
    """
    # Chinese symbols → English
    block = _PUNCT_RE.sub(';', block)
    # Full-width colon to half-width
    block = block.replace('：', ':')
    # Compress tabs/multiple spaces
    block = _SPACE_RE.sub(' ', block)
    # Merge consecutive empty lines
    block = _EOL_RE.sub('\n', block).strip()
    return block

# Placeholder system prompt to solve circular reference issue
# To avoid circular imports, redefine system prompt here
SYSTEM_PROMPT = """
You are a retrieval planning agent. For any user question, you must strictly output 4 tag sections in a fixed order, without missing or combining sections:

<think> … </think>
<search> … </search>
<result> … </result>
<answer> … </answer>

### Rules
1. **<think>** tag should contain reasoning that:
   - First breaks down key terms/concepts in the user question
   - Analyzes which search tool is most appropriate for each keyword
   - Determines search order and dependencies between nodes
   - No word limit, can span multiple paragraphs
2. **<search>** must contain "Nodes:" and "Edges:" sections, with strict format (case-sensitive):
   - Nodes section lists one per line as Label: Query description (Tool)
   - Edges section lists Source label -> Target label, multiple edges separated by semicolons
3. Constraints:
   - Maximum 8 nodes; each query description ≤ 10 words
   - Choose 1 of 3 tools for each node (must be unique per node):
     • **General** = Web search (Google)
     • **News** = News search (GNews)
     • **Academy** = Academic paper search (ArXiv, requires English keywords)
   - No self-loops or cycles; Edges must only refer to declared nodes
4. **<result>** tag will be automatically filled by the system in the **second phase**. **Do not include this tag in the first phase**
5. **<answer>** should provide the final answer based on search results, in any language but complete and accurate
6. No custom tags or Markdown are allowed in the response besides the above tags

### Example

<think>
Question: How do US tariff policies impact 401(k) retirement accounts and prices?

Keywords and tools:
- "US tariffs 2024" → Latest policies → News
- "tariffs impact 401k returns and CPI" → Market/price analysis → General

Dependencies: First get the latest policies, then analyze market and price impacts.
</think>

<search>
Nodes:
A: US tariffs 2024 (News)
B: Tariffs impact on 401k & CPI (General)
C: new US tariff list commodities 2024 (News)
Edges:
A -> B ; C -> B
</search>

<result>
[System will automatically fill this]
</result>

<answer>
Short-term market fluctuations should not cause overreaction; long-term tariffs may reduce corporate profit margins; prices may increase but will take time to reflect in the market; companies face risks in passing costs to consumers; consumers should consider multiple brand options.
</answer>
"""



# 导入检索工具
# from lagent.actions.fin_search import ActionYahooFinance
from lagent.actions.news_search import ActionGNewsAPI
from lagent.actions.web_browser import WebBrowser
from lagent.actions.arxiv_search import ArxivSearch
from lagent.schema import ActionReturn, ActionStatusCode
from lagent.actions.web_browser import GoogleSearch, ContentFetcher # 添加 ContentFetcher

# 使用统一的日志配置
# 删除重复的logger定义

# 正则表达式
THINK_RE = re.compile(r'<think>(.*?)</think>', re.I | re.S)
SEARCH_RE = re.compile(r'<search>(.*?)</search>', re.I | re.S)
RESULT_RE = re.compile(r'<result>(.*?)</result>', re.I | re.S)
ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.I | re.S)
# 更新正则表达式，使其更加宽松
NODE_RE = re.compile(
    r'([A-Z])\s*[:：]\s*'          # 节点标签
    r'(.+?)\s*'                   # 查询文本
    r'[\[(（]\s*([\w\u4e00-\u9fff\- ]+?)\s*[\)）\]]\s*;?$',   # 允许中英混写、空格和破折号
    flags=re.I
)
EDGE_RE = re.compile(r'([A-Z])\s*(?:->|→)\s*([A-Z])', flags=re.I)  # 同时支持->和中文箭头→

# 允许的工具类型
ALLOWED_TOOLS = {'general', 'news', 'academy'}

def get_chat_special_tokens(tok):
    """
    返回 (bos_token, eos_token)；如果模型没有定义就退回默认占位符。
    """
    return tok.bos_token, tok.eos_token

class SearchCache:
    """搜索结果缓存类，将搜索结果持久化到磁盘"""
    
    def __init__(self, cache_dir: str = CACHE_DIR, expiry_days: Optional[int] = CACHE_EXPIRY):
        """初始化缓存
        
        Args:
            cache_dir: 缓存目录
            expiry_days: 缓存过期天数，None表示永不过期
        """
        self.cache_dir = cache_dir
        self.expiry_days = expiry_days
        self.logger = logging.getLogger(__name__)
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 建立各种缓存类型的子目录
        self.cache_types = {
            'general': os.path.join(cache_dir, 'general'),
            'news': os.path.join(cache_dir, 'news'),
            'academy': os.path.join(cache_dir, 'academy')
        }
        
        for cache_type_dir in self.cache_types.values():
            os.makedirs(cache_type_dir, exist_ok=True)
            
        self.logger.info(f"初始化搜索缓存，目录: {cache_dir}，过期天数: {expiry_days}")
        
        # 内存缓存，避免频繁读取磁盘
        self.memory_cache = {}
    
    def _generate_cache_key(self, query: str, tool_type: str) -> str:
        """生成缓存键
        
        Args:
            query: 搜索查询
            tool_type: 工具类型
            
        Returns:
            缓存键
        """
        # 使用查询和工具类型生成哈希值作为缓存键
        hash_key = hashlib.md5((query + tool_type).encode()).hexdigest()
        return hash_key
    
    def _get_cache_file_path(self, cache_key: str, tool_type: str) -> str:
        """获取缓存文件路径
        
        Args:
            cache_key: 缓存键
            tool_type: 工具类型
            
        Returns:
            缓存文件路径
        """
        cache_type_dir = self.cache_types.get(tool_type, self.cache_types['general'])
        return os.path.join(cache_type_dir, f"{cache_key}.pkl")
    
    def get(self, query: str, tool_type: str) -> Optional[str]:
        """获取缓存的搜索结果
        
        Args:
            query: 搜索查询
            tool_type: 工具类型
            
        Returns:
            缓存的搜索结果，如果没有缓存或缓存过期则返回None
        """
        if not CACHE_ENABLED:
            return None
            
        cache_key = self._generate_cache_key(query, tool_type)
        
        # 先检查内存缓存
        if cache_key in self.memory_cache:
            cache_data = self.memory_cache[cache_key]
            if self._is_cache_valid(cache_data):
                self.logger.info(f"从内存缓存获取结果，查询: '{query}'，类型: {tool_type}")
                return cache_data['result']
            else:
                # 内存缓存已过期，删除
                del self.memory_cache[cache_key]
        
        # 检查磁盘缓存
        cache_file = self._get_cache_file_path(cache_key, tool_type)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # 检查缓存是否过期
                if self._is_cache_valid(cache_data):
                    # 添加到内存缓存
                    self.memory_cache[cache_key] = cache_data
                    self.logger.info(f"从磁盘缓存获取结果，查询: '{query}'，类型: {tool_type}")
                    return cache_data['result']
                else:
                    # 缓存过期，删除文件
                    os.remove(cache_file)
                    self.logger.info(f"缓存已过期，删除文件: {cache_file}")
            except Exception as e:
                self.logger.warning(f"读取缓存文件出错: {str(e)}，将删除缓存文件")
                try:
                    os.remove(cache_file)
                except:
                    pass
        
        return None
    
    def set(self, query: str, tool_type: str, result: str) -> None:
        """缓存搜索结果
        
        Args:
            query: 搜索查询
            tool_type: 工具类型
            result: 搜索结果
        """
        if not CACHE_ENABLED:
            return
            
        cache_key = self._generate_cache_key(query, tool_type)
        cache_data = {
            'query': query,
            'tool_type': tool_type,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加到内存缓存
        self.memory_cache[cache_key] = cache_data
        
        # 保存到磁盘
        cache_file = self._get_cache_file_path(cache_key, tool_type)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            self.logger.info(f"保存搜索结果到缓存，查询: '{query}'，类型: {tool_type}")
        except Exception as e:
            self.logger.error(f"保存缓存文件出错: {str(e)}")
    
    def _is_cache_valid(self, cache_data: Dict) -> bool:
        """检查缓存是否有效（未过期）
        
        Args:
            cache_data: 缓存数据
            
        Returns:
            缓存是否有效
        """
        # 如果未设置过期时间，则缓存永不过期
        if self.expiry_days is None:
            return True
            
        # 解析缓存时间戳
        try:
            timestamp = datetime.fromisoformat(cache_data['timestamp'])
            # 计算过期时间
            expiry_time = timestamp + timedelta(days=self.expiry_days)
            # 检查是否过期
            return datetime.now() <= expiry_time
        except Exception as e:
            self.logger.warning(f"检查缓存有效性出错: {str(e)}")
            return False
    
    def clear(self, older_than_days: Optional[int] = None) -> int:
        """清理过期缓存
        
        Args:
            older_than_days: 删除指定天数之前的缓存，None表示使用默认过期天数
            
        Returns:
            删除的缓存文件数量
        """
        if older_than_days is None:
            older_than_days = self.expiry_days
            
        if older_than_days is None:
            self.logger.info("未设置过期时间，不清理缓存")
            return 0
            
        deleted_count = 0
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        
        # 清理内存缓存
        keys_to_delete = []
        for key, cache_data in self.memory_cache.items():
            try:
                timestamp = datetime.fromisoformat(cache_data['timestamp'])
                if timestamp < cutoff_time:
                    keys_to_delete.append(key)
            except:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.memory_cache[key]
        
        # 清理磁盘缓存
        for cache_type, cache_dir in self.cache_types.items():
            if os.path.exists(cache_dir):
                for filename in os.listdir(cache_dir):
                    if not filename.endswith('.pkl'):
                        continue
                        
                    file_path = os.path.join(cache_dir, filename)
                    try:
                        # 获取文件修改时间
                        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if mod_time < cutoff_time:
                            os.remove(file_path)
                            deleted_count += 1
                    except Exception as e:
                        self.logger.warning(f"清理缓存文件出错: {str(e)}")
        
        self.logger.info(f"清理了 {deleted_count} 个过期缓存文件")
        return deleted_count

class SearchTools:
    """实现多种搜索工具的封装类"""
    
    def __init__(self, gnews_api_key: str = None, proxy_url: str = PROXY_URL, proxy_dict: dict = PROXY_DICT, 
                 google_api_key: str = None, batch_search_delay: float = 0.05, topk: int = SEARCH_TOPK,
                 use_cache: bool = CACHE_ENABLED, cache_dir: str = CACHE_DIR, cache_expiry: int = CACHE_EXPIRY,
                 fetch_full_content: bool = FETCH_FULL_CONTENT):
        """初始化各种搜索工具
        
        Args:
            gnews_api_key: GNews API密钥，如果为None则不初始化新闻搜索
            proxy_url: 代理URL，用于避开API限制
            proxy_dict: 代理字典，用于requests和httpx库
            google_api_key: Google Serper API密钥，用于GoogleSearch
            batch_search_delay: 批量搜索时每个查询之间的延迟（秒），设为0则无延迟
            topk: 每个搜索返回的结果数量，默认使用全局设置的SEARCH_TOPK
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
            cache_expiry: 缓存过期天数
            fetch_full_content: 是否获取搜索结果的完整网页内容
        """
        # 保存代理URL和代理字典
        self.proxy_url = proxy_url
        self.proxy_dict = proxy_dict
        self.batch_search_delay = batch_search_delay # 新增
        self.topk = topk # 新增
        self.fetch_full_content = fetch_full_content # 新增
        
        # 初始化缓存
        self.use_cache = use_cache # 新增
        if use_cache:
            self.cache = SearchCache(cache_dir, cache_expiry)
        else:
            self.cache = None
        
        # 初始化新闻搜索工具（如果有API密钥）
        self.news_tool = None
        if gnews_api_key:
            self.news_tool = ActionGNewsAPI(api_key=gnews_api_key)
        
        # 初始化网页搜索工具（使用正确的代理参数）
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化WebBrowser，配置代理: proxies={proxy_dict}, proxy={proxy_url}")
        self.web_tool = WebBrowser(
            searcher_type='DuckDuckGoSearch', 
            topk=self.topk, # 修改为使用self.topk
            proxies=proxy_dict,    # 用于requests和httpx
            proxy=proxy_url,       # 用于aiohttp
            timeout=10
        )
        
        # 初始化Google搜索工具（不使用代理）
        self.google_search = None
        if google_api_key:
            self.logger.info(f"初始化GoogleSearch，API密钥: {google_api_key}")
            self.google_search = GoogleSearch(api_key=google_api_key, topk=self.topk) # 使用self.topk
            # 同时创建一个专门用于新闻搜索的GoogleSearch实例
            self.google_news = GoogleSearch(api_key=google_api_key, topk=self.topk, search_type='news') # 使用self.topk
        
        # 初始化学术搜索工具（不传递代理参数）
        self.academy_tool = ArxivSearch(
            top_k_results=2 # 修改为1，与search_managertest.py一致，或根据需要调整为self.topk
            # max_results=5
        )

        # 初始化内容获取器（如果需要获取完整内容）
        if fetch_full_content: # 新增
            self.content_fetcher = ContentFetcher(
                timeout=FETCH_TIMEOUT,
                proxies=proxy_dict,
                proxy=proxy_url
            )
        else:
            self.content_fetcher = None
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)

        # 缓存命中统计
        self.cache_stats = { # 新增
            'hits': 0,
            'misses': 0
        }
    
    @retry(stop=(stop_after_attempt(MAX_RETRIES) | stop_after_delay(SEARCH_TIMEOUT)),
           wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
           reraise=False)
    def _search_with_retry(self, search_func, query, tool_name="搜索"):
        """使用重试机制包装搜索函数
        
        Args:
            search_func: 要调用的搜索函数
            query: 搜索查询
            tool_name: 工具名称（用于日志）
            
        Returns:
            搜索结果或错误信息
        """
        try:
            # 设置超时
            result = search_func(query)
            return result
        except Exception as e:
            self.logger.warning(f"{tool_name}失败，准备重试: {str(e)}")
            raise TryAgain(f"{tool_name}失败: {str(e)}")
    
    def _guess_lang(self, text: str) -> str:
        """非常粗糙的中英判断：出现任一汉字 → zh，否则 en"""
        return "zh" if re.search(r'[\u4e00-\u9fff]', text) else "en"
    
    def search_news(self, query: str) -> str:
        """进行新闻搜索"""
        # 构建实际查询，根据设置添加日期范围或时间后缀
        actual_query = query
        
        # 添加日期范围（优先于时间后缀）
        if USE_DATE_RANGE and "news" in DATE_RANGE_FOR_TYPES:
            # 检查查询中是否已经包含日期范围参数
            if not any(x in query for x in ["after:", "before:"]):
                # 使用Google日期范围语法
                date_range = f" after:{DATE_RANGE_START} before:{DATE_RANGE_END}"
                actual_query = query + date_range
                self.logger.info(f"添加日期范围到新闻搜索: '{date_range}'")
        # 如果未使用日期范围但启用了时间后缀
        elif ADD_TIME_TO_SEARCH:
            # 检查是否已经包含时间信息
            if not any(substr in query for substr in [TIME_SUFFIX_ZH, TIME_SUFFIX_EN]):
                lang = self._guess_lang(query)
                if lang == "zh":
                    actual_query = f"{query} {TIME_SUFFIX_ZH}"
                else:
                    actual_query = f"{query} {TIME_SUFFIX_EN}"
        
        # 检查缓存 - 使用带时间信息的实际查询作为缓存键
        if self.use_cache:
            cached_result = self.cache.get(actual_query, 'news')
            if cached_result:
                self.cache_stats['hits'] += 1
                return cached_result
            self.cache_stats['misses'] += 1

        # 首先检查是否有GNews API工具
        if self.news_tool:
            try:
                def _do_search(q):
                    # 自动检测语言
                    lang = self._guess_lang(q)
                    # 根据语言设置相应的country
                    country = "cn" if lang == "zh" else "us"
                    # 搜索新闻，修正参数名从lang改为language
                    result = self.news_tool.search_news(q, max_results=1, language=lang, country=country)
                    if result.state == ActionStatusCode.SUCCESS:
                        articles = result.result.get('articles', [])
                        if not articles:
                            self.logger.warning(f"GNews未找到与 {q} 相关的新闻，尝试使用GoogleSearch新闻搜索...")
                            # 如果有GoogleSearch，尝试使用它进行新闻搜索
                            if self.google_news:
                                return self._try_google_news_search(q)
                            # 否则回退到网页搜索
                            return self.search_web(q)
                        
                        news_results = f"关于 {q} 的新闻:\n"
                        # 修正：处理最多3条新闻，并应用_truncate
                        for i, article in enumerate(articles, 1): # 保持最多3条
                            title  = _truncate(article.get('title', 'N/A')) # 应用截断
                            source = article.get('source', {}).get('name', 'N/A')
                            date   = article.get('publishedAt', 'N/A')[:10] # 只取日期部分
                            desc   = _truncate(article.get('description', 'N/A')) # 应用截断
                            # 采用用户提供的格式
                            news_results += f'• "{title}" — {source} ({date})\n  {desc}\n'
                        
                        return news_results
                    
                    self.logger.warning(f"无法找到与 {q} 相关的新闻，尝试使用GoogleSearch新闻搜索...")
                    # 如果有GoogleSearch，尝试使用它进行新闻搜索
                    if self.google_news:
                        return self._try_google_news_search(q)
                    # 否则回退到网页搜索
                    return self.search_web(q)
                
                return self._search_with_retry(_do_search, actual_query, "GNews新闻搜索")
            except RetryError as e:
                self.logger.error(f"GNews新闻搜索失败，已达到最大重试次数: {str(e)}")
                # 如果有GoogleSearch，尝试使用它进行新闻搜索
                if self.google_news:
                    self.logger.info(f"尝试使用GoogleSearch新闻搜索...")
                    return self._try_google_news_search(actual_query)
                # 否则回退到网页搜索
                self.logger.info(f"尝试使用网页搜索作为回退...")
                return self.search_web(actual_query)
            except Exception as e:
                self.logger.error(f"GNews新闻搜索出错: {str(e)}")
                # 如果有GoogleSearch，尝试使用它进行新闻搜索
                if self.google_news:
                    self.logger.info(f"尝试使用GoogleSearch新闻搜索...")
                    return self._try_google_news_search(actual_query)
                # 否则回退到网页搜索
                self.logger.info(f"尝试使用网页搜索作为回退...")
                return self.search_web(actual_query)
        elif self.google_news:
            # 如果没有GNews API但有GoogleSearch，尝试使用它进行新闻搜索
            return self._try_google_news_search(actual_query)
        else:
            # 如果两者都没有，回退到通用搜索
            return f"新闻搜索功能未启用，回退到通用搜索...\n" + self.search_web(actual_query)
    
    def _try_google_news_search(self, query: str) -> str:
        """尝试使用GoogleSearch的news模式进行搜索"""
        # 构建实际查询，根据设置添加日期范围或时间后缀
        actual_query = query
        
        # 添加日期范围（优先于时间后缀）
        if USE_DATE_RANGE and "news" in DATE_RANGE_FOR_TYPES:
            # 检查查询中是否已经包含日期范围参数
            if not any(x in query for x in ["after:", "before:"]):
                # 使用Google日期范围语法
                date_range = f" after:{DATE_RANGE_START} before:{DATE_RANGE_END}"
                actual_query = query + date_range
                self.logger.info(f"添加日期范围到GoogleNews搜索: '{date_range}'")
        # 如果未使用日期范围但启用了时间后缀
        elif ADD_TIME_TO_SEARCH:
            # 检查是否已经包含时间信息
            if not any(substr in query for substr in [TIME_SUFFIX_ZH, TIME_SUFFIX_EN]):
                lang = self._guess_lang(query)
                if lang == "zh":
                    actual_query = f"{query} {TIME_SUFFIX_ZH}"
                else:
                    actual_query = f"{query} {TIME_SUFFIX_EN}"
                
        # 检查缓存 - 使用带时间信息的实际查询作为缓存键
        if self.use_cache:
            cached_result = self.cache.get(actual_query, 'news')
            if cached_result:
                self.cache_stats['hits'] += 1
                return cached_result
            self.cache_stats['misses'] += 1

        try:
            def _do_google_news_search(q):
                # 进行GoogleSearch新闻搜索
                self.logger.info(f"使用GoogleSearch新闻搜索: {q}")
                search_results = self.google_news.search(q)
                if not search_results:
                    self.logger.warning(f"GoogleSearch新闻搜索未找到结果，回退到通用搜索")
                    return self.search_web(q)
                
                news_results = f"关于 {q} 的新闻:\n"

                # 如果启用了获取完整内容功能，预先获取前N个结果的完整内容
                full_contents = {}
                if self.fetch_full_content and self.content_fetcher:
                    # 收集前FETCH_CONTENT_TOPK个结果的URL
                    urls_to_fetch = []
                    for idx, item in list(search_results.items())[:FETCH_CONTENT_TOPK]:
                        if 'url' in item and item['url']:
                            urls_to_fetch.append((idx, item['url']))
                    
                    # 获取完整内容
                    self.logger.info(f"获取 {len(urls_to_fetch)} 个新闻页面的完整内容")
                    for idx, url in urls_to_fetch:
                        try:
                            success, content = self.content_fetcher.fetch(url)
                            if success and content:
                                full_contents[idx] = content
                                self.logger.info(f"成功获取新闻 URL {url} 的完整内容，长度: {len(content)} 字符")
                            else:
                                self.logger.warning(f"无法获取新闻 URL {url} 的完整内容: {content}")
                        except Exception as e:
                            self.logger.error(f"获取新闻 URL {url} 内容时出错: {str(e)}")
                
                for idx, item in search_results.items():
                    # 获取标题
                    title = _truncate(item.get('title', 'N/A'), MAX_SNIPPET//4) # 使用 MAX_SNIPPET
                    news_results += f"• \"{title}\""
                    
                    # 获取source（来源）
                    source = item.get('source', {}).get('name', '')
                    
                    # 获取发布日期
                    date = item.get('publishedAt', '')[:10] if item.get('publishedAt') else ''
                    
                    # 添加来源和日期信息（如果有）
                    source_info = ""
                    if source or date:
                        source_info = f" — {source}" if source else ""
                        source_info += f" ({date})" if date else ""
                        news_results += f"{source_info}"
                    
                    news_results += "\n"
                    
                    # 获取并合并所有可能的内容源
                    contents = []
                    
                    # 如果有完整内容，使用它
                    if idx in full_contents:
                        # 将完整内容添加到内容列表的开头（优先使用）
                        full_text = full_contents[idx]
                        # 提取前1000个字符作为摘要（避免内容过长）
                        contents.append(full_text[:1000])
                    
                    # 获取summ（主要摘要）
                    summ = item.get('summ', '')
                    if summ:
                        contents.append(summ)
                    
                    # 获取snippet（如果存在且与summ不同）
                    snippet = item.get('snippet', '')
                    if snippet and snippet != summ:
                        contents.append(snippet)
                    
                    # 获取description（如果存在且与前两者不同）
                    description = item.get('description', '')
                    if description and description not in [summ, snippet]:
                        contents.append(description)
                    
                    # 合并内容，确保总长度不超过MAX_SNIPPET
                    combined_content = " ".join(contents)
                    content = _truncate(combined_content, MAX_SNIPPET) # 使用 MAX_SNIPPET
                    
                    news_results += f"  {content}\n"
                
                return news_results
            
            result = self._search_with_retry(_do_google_news_search, actual_query, "GoogleSearch新闻搜索")
            # 缓存结果使用实际查询作为键
            if self.use_cache and self._is_successful_search_result(result, actual_query):
                self.cache.set(actual_query, 'news', result)
            return result
        except Exception as e:
            self.logger.error(f"GoogleSearch新闻搜索出错: {str(e)}")
            return self.search_web(actual_query)
    
    def search_web(self, query: str) -> str:
        """进行网页搜索"""
        # 构建实际查询，根据设置添加日期范围或时间后缀
        actual_query = query
        
        # 添加日期范围（优先于时间后缀）
        if USE_DATE_RANGE and "general" in DATE_RANGE_FOR_TYPES:
            # 检查查询中是否已经包含日期范围参数
            if not any(x in query for x in ["after:", "before:"]):
                # 使用Google日期范围语法
                date_range = f" after:{DATE_RANGE_START} before:{DATE_RANGE_END}"
                actual_query = query + date_range
                self.logger.info(f"添加日期范围到通用搜索: '{date_range}'")
        # 如果未使用日期范围但启用了时间后缀
        elif ADD_TIME_TO_SEARCH:
            # 检查是否已经包含时间信息
            if not any(substr in query for substr in [TIME_SUFFIX_ZH, TIME_SUFFIX_EN]):
                lang = self._guess_lang(query)
                if lang == "zh":
                    actual_query = f"{query} {TIME_SUFFIX_ZH}"
                else:
                    actual_query = f"{query} {TIME_SUFFIX_EN}"
                
        # 检查缓存 - 使用带时间信息的实际查询作为缓存键
        if self.use_cache:
            cached_result = self.cache.get(actual_query, 'general')
            if cached_result:
                self.cache_stats['hits'] += 1
                return cached_result
            self.cache_stats['misses'] += 1

        # 首先尝试使用GoogleSearch进行搜索
        if self.google_search:
            try:
                def _do_google_search(q):
                    # 使用GoogleSearch进行网页搜索
                    self.logger.info(f"使用GoogleSearch进行网页搜索: {q}")
                    search_results = self.google_search.search(q)
                    if not search_results:
                        self.logger.warning(f"GoogleSearch未找到与 {q} 相关的网页信息，尝试使用DuckDuckGo搜索")
                        # 回退到原来的DuckDuckGo搜索
                        return self._try_duckduckgo_search(q)
                    
                    web_results = f"关于 {q} 的网页搜索结果:\n"

                    # 如果启用了获取完整内容功能，预先获取前N个结果的完整内容
                    full_contents = {}
                    if self.fetch_full_content and self.content_fetcher:
                        # 收集前FETCH_CONTENT_TOPK个结果的URL
                        urls_to_fetch = []
                        for idx, item in list(search_results.items())[:FETCH_CONTENT_TOPK]:
                            if 'url' in item and item['url']:
                                urls_to_fetch.append((idx, item['url']))
                        
                        # 获取完整内容
                        self.logger.info(f"获取 {len(urls_to_fetch)} 个网页的完整内容")
                        for idx, url in urls_to_fetch:
                            try:
                                success, content = self.content_fetcher.fetch(url)
                                if success and content:
                                    full_contents[idx] = content
                                    self.logger.info(f"成功获取 URL {url} 的完整内容，长度: {len(content)} 字符")
                                else:
                                    self.logger.warning(f"无法获取 URL {url} 的完整内容: {content}")
                            except Exception as e:
                                self.logger.error(f"获取 URL {url} 内容时出错: {str(e)}")
                    
                    # 处理所有搜索结果
                    for idx, item in search_results.items():
                        web_results += f"结果 {idx+1}:\n"
                        
                        # 获取标题
                        title = _truncate(item.get('title', 'N/A'), MAX_SNIPPET//4) # 使用 MAX_SNIPPET
                        web_results += f"标题: {title}\n"
                        
                        # 获取并合并所有可能的内容源
                        contents = []
                        
                        # 如果有完整内容，使用它
                        if idx in full_contents:
                            # 将完整内容添加到内容列表的开头（优先使用）
                            full_text = full_contents[idx]
                            # 提取前1000个字符作为摘要（避免内容过长）
                            contents.append(full_text[:1000])
                        
                        # 获取summ（主要摘要）
                        summ = item.get('summ', '')
                        if summ:
                            contents.append(summ)
                        
                        # 获取snippet（如果存在且与summ不同）
                        snippet = item.get('snippet', '')
                        if snippet and snippet != summ:
                            contents.append(snippet)
                        
                        # 获取description（如果存在且与前两者不同）
                        description = item.get('description', '')
                        if description and description not in [summ, snippet]:
                            contents.append(description)
                        
                        # 合并内容，确保总长度不超过MAX_SNIPPET
                        combined_content = " ".join(contents)
                        content = _truncate(combined_content, MAX_SNIPPET) # 使用 MAX_SNIPPET
                        
                        web_results += f"摘要: {content}\n\n"
                    
                    return web_results
                
                result = self._search_with_retry(_do_google_search, actual_query, "GoogleSearch网页搜索")
                # 缓存结果使用实际查询作为键
                if self.use_cache and self._is_successful_search_result(result, actual_query):
                    self.cache.set(actual_query, 'general', result)
                return result
            except Exception as e:
                self.logger.error(f"GoogleSearch网页搜索出错: {str(e)}")
                # 回退到原来的DuckDuckGo搜索
                self.logger.info(f"尝试使用DuckDuckGo搜索作为回退...")
                return self._try_duckduckgo_search(actual_query)
        else:
            # 如果没有GoogleSearch，使用原来的DuckDuckGo搜索
            return self._try_duckduckgo_search(actual_query)
    
    def _try_duckduckgo_search(self, query: str) -> str:
        """使用原来的DuckDuckGo进行搜索"""
        # 构建实际查询，根据设置添加时间信息
        actual_query = query
        
        # DuckDuckGo不支持Google的日期范围语法，仅使用时间后缀
        if ADD_TIME_TO_SEARCH and not any(substr in query for substr in [TIME_SUFFIX_ZH, TIME_SUFFIX_EN]):
            lang = self._guess_lang(query)
            if lang == "zh":
                actual_query = f"{query} {TIME_SUFFIX_ZH}"
            else:
                actual_query = f"{query} {TIME_SUFFIX_EN}"
                
        # 检查缓存 - 使用带时间信息的实际查询作为缓存键
        if self.use_cache:
            cached_result = self.cache.get(actual_query, 'general')
            if cached_result:
                self.cache_stats['hits'] += 1
                return cached_result
            self.cache_stats['misses'] += 1

        try:
            def _do_search(q):
                # 记录代理使用信息
                if self.proxy_dict:
                    self.logger.info(f"DuckDuckGo搜索使用代理: {self.proxy_url}")
                
                # 进行网页搜索, 使用初始化时设置的 topk (默认为2)
                search_results = self.web_tool.search(q) 
                if not search_results:
                    return f"未找到与 {q} 相关的网页信息"
                
                web_results = f"关于 {q} 的网页搜索结果:\n"
                # 修正：处理所有返回的结果，并应用_truncate
                for idx, item in search_results.items():
                    web_results += f"结果 {idx+1}:\n"
                    title = _truncate(item.get('title', 'N/A')) # 应用截断
                    web_results += f"标题: {title}\n"
                    # web_results += f"URL: {item.get('url', 'N/A')}\n"
                    summ = _truncate(item.get('summ', 'N/A')) # 应用截断
                    web_results += f"摘要: {summ}\n\n"
                
                return web_results
            
            return self._search_with_retry(_do_search, query, "DuckDuckGo搜索")
        except RetryError as e:
            # self.logger.error(f"DuckDuckGo搜索失败，已达到最大重试次数: {str(e)}")
            self.logger.error(f"DuckDuckGo搜索失败，已达到最大重试次数")
            return f"网页搜索失败。无法获取与 '{actual_query}' 相关的网页信息。"
        except Exception as e:
            self.logger.error(f"DuckDuckGo搜索出错")
            return f"网页搜索出错"
    
    def search_academy(self, query: str) -> str:
        """进行学术论文搜索"""
        # 构建实际查询，根据设置添加日期范围或时间后缀
        actual_query = query
        
        # 添加日期范围（优先于时间后缀）- 仅在学术搜索类型被包含时添加
        if USE_DATE_RANGE and "academy" in DATE_RANGE_FOR_TYPES:
            # 检查查询中是否已经包含日期范围参数
            if not any(x in query for x in ["after:", "before:"]):
                # ArXiv不直接支持after:before:语法，但我们仍然为搜索缓存键添加它们
                date_range = f" after:{DATE_RANGE_START} before:{DATE_RANGE_END}"
                actual_query = query + date_range
                self.logger.info(f"添加日期范围到学术搜索缓存键: '{date_range}'")
                # 注意：实际搜索时仍使用原始查询，因为ArXiv API不支持此语法
        
        # 检查缓存
        if self.use_cache:
            cached_result = self.cache.get(actual_query, 'academy')
            if cached_result:
                self.cache_stats['hits'] += 1
                return cached_result
            self.cache_stats['misses'] += 1

        try:
            def _do_search(q):
                # 在学术搜索中使用原始查询，因为ArXiv API可能不支持Google日期语法
                original_query_for_arxiv = query if USE_DATE_RANGE and "academy" in DATE_RANGE_FOR_TYPES else q

                # 使用查询搜索学术论文
                result_obj = self.academy_tool.get_arxiv_article_information(original_query_for_arxiv)
                if isinstance(result_obj, dict) and 'content' in result_obj:
                    content = result_obj['content']
                    if "没有找到相关的arxiv文章" in content or not content.strip():
                        self.logger.warning(f"未找到与 {original_query_for_arxiv} 相关的学术论文，尝试使用网页搜索作为回退...")
                        return self.search_web(q)
                    # 应用_truncate函数限制结果长度
                    truncated_content = _truncate(content)
                    return f"关于 {original_query_for_arxiv} 的学术论文:\n{truncated_content}"
                
                self.logger.warning(f"未找到与 {original_query_for_arxiv} 相关的学术论文，尝试使用网页搜索作为回退...")
                return self.search_web(q)
            
            result = self._search_with_retry(_do_search, actual_query, "学术搜索")
            # 缓存结果使用带日期范围的实际查询作为键
            if self.use_cache and self._is_successful_search_result(result, actual_query):
                self.cache.set(actual_query, 'academy', result)
            return result
        except RetryError as e:
            self.logger.error(f"学术搜索失败，已达到最大重试次数: {str(e)}")
            self.logger.info(f"尝试使用网页搜索作为回退...")
            return self.search_web(actual_query)
        except Exception as e:
            self.logger.error(f"学术搜索出错: {str(e)}")
            self.logger.info(f"尝试使用网页搜索作为回退...")
            return self.search_web(actual_query)
    
    def search_by_type(self, query: str, node_type: str) -> str:
        """根据节点类型选择不同的搜索工具"""
        node_type = node_type.lower().strip()
        
        # 记录代理使用信息
        if self.proxy_url: # 修正：应为 self.proxy_url 或 self.proxy_dict
            self.logger.info(f"使用代理执行搜索: 类型={node_type}, 查询={query}")
        
        # 消融测试：如果启用了USE_GENERAL_FOR_ALL，所有搜索都使用general
        if USE_GENERAL_FOR_ALL: # 新增
            self.logger.info(f"消融测试模式：强制使用通用搜索，原始类型: {node_type}, 查询: {query}")
            return self.search_web(query)

        # 映射工具名称
        mapping = {
            'search': 'general',   # 宽容写法
            '新闻': 'news',
            '学术': 'academy'
        }
        node_type = mapping.get(node_type, node_type) # 保持原样，若无匹配则使用原始node_type
        
        if node_type == 'news':
            return self.search_news(query)
        elif node_type == 'academy':
            # 根据全局参数决定使用哪种搜索方式
            if USE_GENERAL_FOR_ACADEMY: # 新增
                self.logger.info(f"使用通用搜索引擎搜索学术内容: {query}")
                return self.search_web(query)
            else:
                return self.search_academy(query)
        else:  # 默认使用general (包括未在mapping中定义的node_type)
            return self.search_web(query)

    def batch_search(self, queries: List[Dict]) -> List[str]:
        """批量处理搜索请求，优化并行处理
        
        Args:
            queries: 搜索请求列表，每个请求包含 'query' 和 'type' 键
            
        Returns:
            搜索结果列表
        """
        results = []
        
        # 按类型分组查询，便于批量处理
        type_grouped_queries = defaultdict(list)
        query_indices = {}
        
        for i, query_dict in enumerate(queries):
            query_type = query_dict.get('type', 'general').lower()
            query_text = query_dict.get('query', '')
            
            type_grouped_queries[query_type].append(query_text)
            query_indices[(query_type, len(type_grouped_queries[query_type])-1)] = i
        
        # 结果列表预分配
        results = [""] * len(queries)
        
        # 处理不同类型的查询
        for query_type, type_queries in type_grouped_queries.items():
            for i, query in enumerate(type_queries):
                try:
                    # 添加随机延迟，避免并发请求过多导致API封禁
                    if self.batch_search_delay > 0: # 使用self.batch_search_delay
                        # 使用配置的延迟，可以加入少量随机性避免完全同步
                        time.sleep(random.uniform(self.batch_search_delay * 0.8, self.batch_search_delay * 1.2))
                    
                    result = self.search_by_type(query, query_type)
                    original_idx = query_indices.get((query_type, i), -1)
                    if original_idx >= 0:
                        results[original_idx] = result
                except Exception as e:
                    self.logger.error(f"批量搜索处理查询时出错: {str(e)}")
                    original_idx = query_indices.get((query_type, i), -1)
                    if original_idx >= 0:
                        results[original_idx] = f"搜索过程出错: {str(e)}"
        
        return results

    def get_cache_stats(self) -> Dict[str, int]: # 新增
        """获取缓存命中统计
        
        Returns:
            缓存命中统计字典
        """
        if not self.use_cache or self.cache is None:
            return { 'hits': 0, 'misses': 0, 'total': 0, 'hit_rate': 0}

        total = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = 0 if total == 0 else (self.cache_stats['hits'] / total) * 100
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'total': total,
            'hit_rate': hit_rate
        }
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> int: # 新增
        """清理缓存
        
        Args:
            older_than_days: 删除指定天数之前的缓存
            
        Returns:
            删除的缓存文件数量
        """
        if not self.use_cache or self.cache is None:
            return 0
            
        return self.cache.clear(older_than_days)

    def _is_successful_search_result(self, result_text: str, query_for_log: str) -> bool:
        """检查搜索结果是否表示成功，以便决定是否缓存"""
        if not result_text:
            self.logger.info(f"搜索 '{query_for_log}' 返回空结果，不缓存。")
            return False

        # 已知的特定错误消息或模式
        error_indicators = [
            f"未找到与 {query_for_log} 相关的网页信息", # DuckDuckGo specific
            "网页搜索失败。",                       # DuckDuckGo specific prefix
            "网页搜索出错",                         # DuckDuckGo specific
            "没有找到相关的arxiv文章",              # Arxiv specific
            "功能未启用",                         # General
            "搜索失败",                           # General
            "搜索出错",                           # General
            "无法找到与",                          # General prefix
            "未找到与",                           # General prefix
        ]

        # 检查是否有完全匹配的错误信息（考虑 query 本身可能包含在错误信息中）
        # 比如 DuckDuckGo 的 "未找到与 '{query}' 相关的网页信息"
        # (注意：上面 error_indicators 中已包含一个 query_for_log 的格式化版本)

        for indicator in error_indicators:
            if indicator in result_text:
                # 使用 _truncate 避免日志过长
                # _truncate 是全局函数，这里直接使用
                self.logger.info(f"搜索 '{query_for_log}' 的结果包含错误指示: '{indicator}'. 结果预览: '{_truncate(result_text, 200)}'. 不缓存.")
                return False
        
        # 如果没有检测到明确的错误指示，则认为成功
        return True

class DAGParser:
    """解析DAG结构并执行搜索计划"""
    
    def __init__(self, search_tools: SearchTools, max_nodes: int = 6):
        """初始化DAG解析器
        
        Args:
            search_tools: 搜索工具
            max_nodes: 最大解析节点数量，默认为10
        """
        self.search_tools = search_tools
        self.max_nodes = max_nodes
        self.logger = logging.getLogger(__name__)
    
    def parse_dag(self, dag_text: str) -> Tuple[Dict[str, Dict], List[Tuple[str, str]]]:
        """解析DAG文本，提取节点和边"""
        # 先进行格式归一化处理
        dag_text = canonicalize_search_block(dag_text)

        nodes = {}
        edges = []
        lines = dag_text.splitlines()
        in_edge = False
        
        for raw in lines:
            ln = raw.strip()
            if not ln:
                continue

            # 1) 进入 Edges 部分后，整行都按 Edge 解析
            if ln.lower().startswith('edges'):
                in_edge = True
                ln = ln[5:]  # 剪掉 'Edges' 字样自身，剩下半行也扫一遍
                # fallthrough：仍然让下面 Edge 逻辑处理
            
            if in_edge:
                for edge_part in ln.split(';'):
                    m_edge = EDGE_RE.search(edge_part)
                    if m_edge:
                        src, tgt = m_edge.groups()
                        edges.append((src.upper(), tgt.upper()))  # 统一转为大写 
                continue

            # 2) Nodes 部分
            m_node = NODE_RE.search(ln)
            # m_node = NODE_RE.match(ln)
            if m_node:
                label, query, node_type = m_node.groups()
                label = label.upper()  # 统一转为大写
                # 检查是否已达到最大节点数限制
                if len(nodes) >= self.max_nodes:
                    self.logger.warning(f"达到最大节点数限制({self.max_nodes})，忽略额外节点")
                    continue
                
                # 规范化工具类型
                node_type = node_type.lower().strip()
                mapping = {
                    'search': 'general',   # 宽容写法
                    '新闻': 'news',
                    '学术': 'academy'
                }
                node_type = mapping.get(node_type, node_type)
                
                nodes[label] = {"query": query.strip(), "type": node_type}
        
        # 记录解析结果
        self.logger.info(f"DAG解析结果: 节点={list(nodes.keys())}, 边={edges}")
        return nodes, edges
    
    def execute_dag(self, dag_text: str) -> str:
        """执行DAG搜索计划"""
        try:
            nodes, edges = self.parse_dag(dag_text)
            
            # 记录所有节点信息
            self.logger.info(f"DAG解析结果: 节点数量 {len(nodes)} (最大限制: {self.max_nodes}), 边数量 {len(edges)}")
            for label, info in nodes.items():
                self.logger.info(f"节点 {label}: 查询='{info['query']}', 类型={info['type']}")
            
            # 记录所有边信息
            for source, target in edges:
                self.logger.info(f"边: {source} -> {target}")
            
            # 构建依赖图和入度表
            graph = defaultdict(list)
            in_degree = {node: 0 for node in nodes}
            
            for source, target in edges:
                graph[source].append(target)
                in_degree[target] = in_degree.get(target, 0) + 1
            
            # 拓扑排序确定执行顺序
            queue = deque([node for node in nodes if in_degree[node] == 0])
            execution_order = []
            
            while queue:
                node = queue.popleft()
                execution_order.append(node)
                
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            # 记录执行顺序
            self.logger.info(f"节点执行顺序: {' -> '.join(execution_order)}")
            
            # 批量准备查询
            search_queries = []
            node_to_idx = {}
            
            for i, node_label in enumerate(execution_order):
                node_info = nodes[node_label]
                search_queries.append({
                    'query': node_info['query'],
                    'type': node_info['type']
                })
                node_to_idx[node_label] = i
            
            # 添加搜索超时控制
            search_start_time = time.time()
            overall_timeout = SEARCH_TIMEOUT * 2  # 整体超时时间为单个搜索超时的两倍
            
            try:
                # 批量执行查询，设置总体超时
                search_results = self.search_tools.batch_search(search_queries)
                search_duration = time.time() - search_start_time
                
                if search_duration > overall_timeout:
                    self.logger.warning(f"搜索执行时间 {search_duration:.2f}秒 超过设定阈值 {overall_timeout}秒")
            except Exception as e:
                self.logger.error(f"执行DAG搜索计划时批量查询出错: {str(e)}")
                search_results = [f"搜索执行出错: {str(e)}"] * len(search_queries)
            
            # 组织结果
            node_results = {}
            all_results = []
            
            for node_label in execution_order:
                idx = node_to_idx[node_label]
                # 检查是否有结果，如果索引越界或结果为空，提供友好提示
                if idx < len(search_results) and search_results[idx]:
                    result = search_results[idx]
                else:
                    result = f"节点 {node_label} 的搜索结果不可用。"
                    
                node_type = nodes[node_label]['type']
                
                node_results[node_label] = result
                all_results.append(f"节点 {node_label} ({node_type}) 搜索结果:\n{result}\n")
                
                # 记录单个节点的搜索结果
                result_preview = result[:200] + "..." if len(result) > 200 else result
                # self.logger.info(f"节点 {node_label} ({node_type}) 搜索结果:\n{result_preview}")
            
            return "\n".join(all_results)
        except Exception as e:
            self.logger.error(f"执行DAG搜索计划时出错: {str(e)}")
            return f"执行DAG搜索计划时出错: {str(e)}\n\n请检查搜索图结构是否正确，或稍后重试。"

def check_r1(text: str, first_stage=False) -> bool:
    """Check if output conforms to R1 template format
    
    Args:
        text: Text to check
        first_stage: Whether it's first-stage generation, if so, only check think and search parts
    """
    if first_stage:
        # First stage only checks think and search parts
        pattern = re.compile(
            r'<think>[\s\S]+?</think>'
            r'[\s\S]*?<search>[\s\S]+?</search>',
            re.IGNORECASE
        )
    else:
        # Complete generation checks all parts
        pattern = re.compile(
            r'<think>[\s\S]+?</think>'
            r'[\s\S]*?<search>[\s\S]+?</search>'
            r'[\s\S]*?<result>[\s\S]*?</result>'  
            r'[\s\S]*?<answer>[\s\S]+?</answer>',
            re.IGNORECASE
        )
    return pattern.search(text) is not None

def check_dag(full_output: str, max_nodes=6) -> bool:
    """检查DAG格式是否正确
    
    Args:
        full_output: 完整输出文本
        max_nodes: 最大节点数量限制，默认为6
    """
    m = SEARCH_RE.search(full_output)
    if not m:
        return False
    dag_text = m.group(1).strip()
    
    try:
        # 创建一个仅用于解析的DAGParser
        parser = DAGParser(search_tools=None, max_nodes=max_nodes)
        nodes, edges = parser.parse_dag(dag_text)
        
        # ① 节点数限制
        if not (1 <= len(nodes) <= max_nodes):
            logger.warning(f"节点数量 {len(nodes)} 不在允许范围内 (1-{max_nodes})")
            return False
            
        # ② 工具类型合法性检查
        for label, node_info in nodes.items():
            if node_info["type"] not in ALLOWED_TOOLS:
                logger.warning(f"节点 {label} 的工具类型 '{node_info['type']}' 不在允许范围 {ALLOWED_TOOLS}，原文本:\n{dag_text}")
                return False
                
        # ③ 边引用合法性检查
        for src, tgt in edges:
            if src not in nodes or tgt not in nodes or src == tgt:
                # logger.warning(f"边 {src}->{tgt} 包含无效节点或自环，原文本:\n{dag_text}")
                logger.warning(f"边 {src}->{tgt} 包含无效节点或自环")
                return False
                
        # ④ 无环检查 (使用networkx)
        g = nx.DiGraph()
        g.add_nodes_from(nodes.keys())
        g.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(g):
            # logger.warning(f"图包含环路，不是有向无环图(DAG)，原文本:\n{dag_text}")
            logger.warning(f"图包含环路，不是有向无环图(DAG)")
            return False
            
        return True
    except Exception as e:
        # logger.error(f"检查DAG格式时出错: {str(e)}，原文本:\n{dag_text}")
        logger.error(f"检查DAG格式时出错: {str(e)}")
        return False

# 新增：检查第一阶段输出是否合规（包含search标签且DAG格式正确）
def first_stage_valid(text: str) -> bool:
    """检查第一阶段输出是否有效
    
    Args:
        text: 模型输出
        
    Returns:
        bool: 格式是否合法
    """
    # 检查是否包含<search>标签
    search_match = SEARCH_RE.search(text)
    
    if not search_match:
        return False
    
    # 如果第一阶段已经包含<result>标签，认为格式不正确
    result_match = RESULT_RE.search(text)
    if result_match:
        logger.warning("第一阶段输出不应包含<result>标签")
        return False
    
    # 检查DAG格式是否正确
    return check_dag(text)

def safe_chat_template(tokenizer, messages, add_generation_prompt=True):
    """安全地应用聊天模板，处理各种可能的异常情况"""
    
    if not tokenizer.chat_template:
        # raise RuntimeError("当前tokenizer未定义chat_template，已强制要求官方模板，出错立停！")
        raise RuntimeError("当前tokenizer未定义chat_template，已强制要求官方模板，出错立停！")
    try:
        # print(f"DEBUG apply_chat_template >>>\n{messages}\n<<<")

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
    except Exception as e:
        # logging.getLogger(__name__).warning(
        #     f"apply_chat_template error: {e}; use fallback."
        # )
        # === fallback: 使用bos_token和eos_token ===
        # bos_token = tokenizer.bos_token
        # eos_token = tokenizer.eos_token
        # text = bos_token
        # for m in messages:
        #     role = m['role'].capitalize()  # 首字母大写
        #     text += f"{role}\n{m['content']}\n"
        # if add_generation_prompt:
        #     text += f"Assistant\n"
        # return text
        raise RuntimeError(f"apply_chat_template error: {e}; 已强制要求官方模板，出错立停！")

class GenerationManager:
    """管理生成过程，包括中间搜索调用和结果整合"""
    
    def __init__(self, tokenizer, model, search_tools: SearchTools, max_turns=2, max_nodes=6, system_prompt=None):
        """初始化生成管理器
        
        Args:
            tokenizer: 分词器
            model: 语言模型
            search_tools: 搜索工具
            max_turns: 最大搜索轮次
            max_nodes: 最大解析节点数量，默认为8
            system_prompt: 系统提示词，如果为None则使用默认值
        """
        self.tokenizer = tokenizer
        self.model = model
        
        # 保存系统提示词
        global SYSTEM_PROMPT_CN
        self.system_prompt = system_prompt
        if system_prompt is not None:
            SYSTEM_PROMPT_CN = system_prompt  # 更新全局变量以便兼容现有代码
        
        # 确保tokenizer的padding_side设置为'left'，这对解码器模型很重要
        self.tokenizer.padding_side = 'left'
        
        self.search_tools = search_tools
        self.max_turns = max_turns
        self.max_nodes = max_nodes
        self.dag_parser = DAGParser(search_tools, max_nodes=max_nodes)
        # 修正：添加准确的类型注解
        self.trainer: 'SearchAwareGRPOTrainer' = None 
        
        # 新增：存储结构化消息
        self.convs: list[list[dict]] = []
        
    def _to_device(self, batch):
        device = self.trainer.accelerator.device
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    
    # 修正：set_trainer_reference 的类型提示也应修正
    def set_trainer_reference(self, trainer: 'SearchAwareGRPOTrainer'):
        """Sets a reference to the trainer for accessing the accelerator."""
        self.trainer = trainer
    
    def encode_batch(self, add_generation_prompt=True):
        texts = [safe_chat_template(self.tokenizer, c, add_generation_prompt)
                 for c in self.convs]
        return self.tokenizer(texts, return_tensors="pt",
                              padding=True, truncation=True)

    def encode_for_generation(self, convs, add_generation_prompt=True):
        """
        convs 可以是单条对话(list[dict])，也可以是 list[ list[dict] ]。
        """
        if convs and isinstance(convs[0], dict):   # 单条
            convs = [convs]
        texts = [safe_chat_template(self.tokenizer, c, add_generation_prompt)
                 for c in convs]
        return self.tokenizer(texts, return_tensors="pt",
                              padding=True, truncation=True)
    
    def postprocess_predictions(self, predictions: List[str]) -> Tuple[List[str], List[str]]:
        """处理模型预测结果，提取动作和内容
        
        Args:
            predictions: 模型预测文本列表
            
        Returns:
            动作和内容的元组
        """
        actions = []
        contents = []
        
        for prediction in predictions:
            # 检查是否包含搜索标记
            search_match = SEARCH_RE.search(prediction)
            if search_match:
                actions.append('search')
                contents.append(search_match.group(1).strip())
            # 检查是否包含回答标记
            elif ANSWER_RE.search(prediction):
                actions.append('answer')
                contents.append(prediction)
            else:
                actions.append(None)
                contents.append('')
                
        return actions, contents
    
    def execute_predictions(self, predictions: List[str], active_mask: List[bool] = None) -> Tuple[List[str], List[bool], List[bool], List[bool]]:
        """执行预测中的搜索动作，并返回更新后的观察、完成标志和动作有效标志
        
        Args:
            predictions: 模型生成的预测列表
            active_mask: 活跃样本掩码列表，如果为None则假设所有样本都是活跃的
            
        Returns:
            Tuple, 包含以下元素:
            - next_obs: 更新后的观察列表
            - dones: 任务完成标志列表
            - valid_action: 动作有效标志列表 
            - is_search: 是否为搜索动作列表
        """
        batch_size = len(predictions)
        if active_mask is None:
            active_mask = [True] * batch_size
            
        next_obs = [""] * batch_size
        dones = [False] * batch_size
        valid_action = [False] * batch_size
        is_search = [False] * batch_size
        
        for i, (pred, active) in enumerate(zip(predictions, active_mask)):
            if not active:
                continue
                
            # 解析<search>部分
            search_match = SEARCH_RE.search(pred)
            if not search_match:
                if ANSWER_RE.search(pred):
                    # 如果有答案部分，认为任务已完成
                    dones[i] = True
                    valid_action[i] = True
                    is_search[i] = False
                    next_obs[i] = "\n<result>\n[无搜索]\n</result>\n\n"
                    continue
                
            is_search[i] = True
            search_content = search_match.group(1).strip()
                
            # 检查DAG格式是否正确
            dag_ok = check_dag(pred)
            if not dag_ok:
                next_obs[i] = "\n\n<result>\n[搜索结果]\n执行搜索时出错：DAG格式不正确。请检查格式是否符合要求。\n</result>\n\n"
                continue
                
            # 提取DAG结构
            try:
                # 解析DAG内容
                nodes, edges = self.dag_parser.parse_dag(search_content)
                if nodes:
                    # 执行搜索
                    results_text = self.dag_parser.execute_dag(search_content)
                    valid_action[i] = True
                    
                    # 将结果添加到观察，确保格式正确
                    next_obs[i] = f"\n\n<result>\n{results_text}\n</result>\n\n"
                else:
                    # DAG解析失败
                    next_obs[i] = "\n\n<result>\n[搜索结果]\n执行搜索时出错：无法解析搜索图结构。请检查格式是否符合要求。\n</result>\n\n"
            except Exception as e:
                logger.error(f"执行搜索时出错: {str(e)}")
                next_obs[i] = f"\n\n<result>\n[搜索结果]\n执行搜索时出错：{str(e)}\n</result>\n\n"
        
        return next_obs, dones, valid_action, is_search
    
    @torch.no_grad()
    def generate_with_search(self, prompts: List[str], max_new_tokens=512, max_completion_length=None):
        """生成带有搜索功能的回答，包括两阶段生成和搜索执行
        
        Args:
            prompts: 输入提示列表
            max_new_tokens: 最大新token数
            max_completion_length: 最大补全长度
            
        Returns:
            元组，包含 (prompt_parts, completion_parts)，分别是提示部分和完成部分的文本
        """
        logger.info(f"开始 generate_with_search，收到 {len(prompts)} 个 prompts。")

        # Ensure trainer reference is set
        if self.trainer is None:
            raise ValueError("Trainer reference not set in GenerationManager. Call set_trainer_reference.")

        # 使用max_completion_length (如果提供)或回退到max_new_tokens
        max_tokens = max_completion_length if max_completion_length is not None else max_new_tokens
        
        batch_size = len(prompts)
        
        # 初始化活跃状态跟踪
        active_mask = [True] * batch_size
        final_outputs = [""] * batch_size
        first_stage_raw = [""] * batch_size
        prompt_parts = [""] * batch_size      # 新增：分离的提示部分
        completion_parts = [""] * batch_size  # 新增：分离的完成部分
        
        # 使用类中保存的系统提示，而不是导入
        system_prompt = self.system_prompt if self.system_prompt is not None else SYSTEM_PROMPT_CN
        
        # 初始化结构化消息列表（只留 system + user）
        self.convs = [[
            {"role": "system", "content": system_prompt},
            {"role": "user",  "content": prompt},
        ] for prompt in prompts]
        
        # 轮次循环
        for turn in range(self.max_turns):
            if not any(active_mask):
                break
                
            # 过滤活跃样本
            active_indices = [i for i, active in enumerate(active_mask) if active]
            if not active_indices: # Check if there are active indices before logging
                continue

            active_convs = [self.convs[i] for i in active_indices]
            logger.info(f"轮次 {turn+1}/{self.max_turns}: 开始第一阶段生成（搜索计划），活跃样本数: {len(active_indices)}") # Log start of stage 1 gen

            # 第一阶段生成：生成到</search>
            # 批量处理所有活跃输入
            # assistant_skeleton = "<think>\n"
            assistant_skeleton = "<think>\n"
            
            # 准备第一阶段输入
            input_texts = [
                safe_chat_template(
                    self.tokenizer,
                    conv + [{"role": "assistant", "content": assistant_skeleton}],
                    add_generation_prompt=True
                )
                for conv in active_convs
            ]
            
            # 使用_generate方法替代原始的生成逻辑
            first_stage_outputs = self._generate(
                prompts=input_texts,
                max_tokens=max_tokens // 2,
                stop_regex=r"</search>",  # 简化为仅匹配</search>标签
                temperature=0.6,
                top_p=0.9,
            )
            
            # 新增：更新 convs 中的 assistant 内容
            for i, (idx, generated_full_text) in enumerate(zip(active_indices, first_stage_outputs)):
                # Extract what the model actually generated for the assistant part
                # generated_full_text includes the prompt like "System: ... User: ... Assistant: <model_output>"
                # The model is prompted with assistant_skeleton, e.g., "<think>\\n"
                # So, raw_model_output_suffix is what comes AFTER "Assistant: <think>\\n"
                raw_model_output_suffix = generated_full_text.split("<assistant>\\n")[-1] if "<assistant>\\n" in generated_full_text else generated_full_text
                
                # --- START MODIFICATION TO CLEAN assistant output ---
                # Construct the full text as if the model generated it from the start of <think>
                # This means prepending the skeleton part that the model was supposed to continue from.
                text_to_clean = assistant_skeleton + raw_model_output_suffix # assistant_skeleton is e.g. "<think>\\n"
                
                cleaned_full_assistant_message = text_to_clean # Default to the constructed text

                if ENABLE_FORMAT_FIXING: # <-- 新增的全局开关判断
                    think_tag_open_str = "<think>"
                    search_tag_open_str = "<search>"
                    search_tag_close_str = "</search>"
                    
                    # Ensure it effectively starts with <think> (it should due to skeleton)
                    if text_to_clean.strip().lower().startswith(think_tag_open_str):
                        idx_search_close = text_to_clean.lower().rfind(search_tag_close_str)
                        # Find where <think>...</think> block ends using regex
                        think_match_in_full = THINK_RE.search(text_to_clean) # regex: <think>(.*?)</think>

                        if idx_search_close != -1 and think_match_in_full and idx_search_close > think_match_in_full.start():
                            # Found </search> and it's sensibly positioned after start of <think> block.
                            # Truncate text_to_clean at the end of </search>.
                            cleaned_full_assistant_message = text_to_clean[:idx_search_close + len(search_tag_close_str)]
                        else: 
                            # </search> not found, or found in an odd place (e.g., before <think> block ended).
                            # Check if <search> (opening) exists after the <think> block.
                            # Determine search start position: after </think> if possible, otherwise after <think>.
                            search_scan_start_offset = 0
                            if think_match_in_full: # If <think>...</think> is complete
                                search_scan_start_offset = think_match_in_full.end()
                            else: # <think> might be open but not closed by THINK_RE, search after <think> open tag
                                idx_think_open = text_to_clean.lower().find(think_tag_open_str)
                                if idx_think_open != -1:
                                    search_scan_start_offset = idx_think_open + len(think_tag_open_str)
                            
                            idx_search_open = text_to_clean.lower().find(search_tag_open_str, search_scan_start_offset)

                            if idx_search_open != -1:
                                # <search> opened after <think> block (or open think tag), but </search> was not found properly.
                                # Append </search> to the original text_to_clean.
                                cleaned_full_assistant_message = text_to_clean + f"\\n{search_tag_close_str}"
                            elif think_match_in_full: 
                                # <think>...</think> is complete, but no <search> opening found after it.
                                # Add an empty search block after the <think>...</think>.
                                cleaned_full_assistant_message = think_match_in_full.group(0) + f"\\n{search_tag_open_str}\\n{search_tag_close_str}"
                            # else: If <think> didn't close properly (no think_match_in_full) AND no <search> opens after it,
                            # cleaned_full_assistant_message remains text_to_clean (the raw constructed text).
                            # This raw form will then be validated.
                    # else: text_to_clean doesn't start with <think> as expected.
                    # This is highly unlikely given the skeleton. Keep as is (text_to_clean).
                    # Validation will catch this structural issue.
                # --- END MODIFICATION ---

                # Update conversation history
                if not (self.convs[idx] and self.convs[idx][-1]["role"] == "assistant"):
                    self.convs[idx].append({"role": "assistant", "content": ""})
                
                self.convs[idx][-1]["content"] = cleaned_full_assistant_message
                
                first_stage_raw[idx] = cleaned_full_assistant_message

                # IMPORTANT: Modify the item in first_stage_outputs for validation
                # Reconstruct the full text with the cleaned assistant part for validation
                if "<assistant>\\n" in generated_full_text:
                    prompt_part_before_assistant = generated_full_text.split("<assistant>\\n")[0] + "<assistant>\\n"
                    first_stage_outputs[i] = prompt_part_before_assistant + cleaned_full_assistant_message
                else: # Fallback, should not happen with add_generation_prompt=True and assistant_skeleton
                    first_stage_outputs[i] = cleaned_full_assistant_message 
            
            # 新增：检测格式，不合规直接标记为done 
            for i, (idx, out) in enumerate(zip(active_indices, first_stage_outputs)): # out is now the modified version
                # 使用first_stage_valid函数进行格式检查
                valid = first_stage_valid(out)
                if not valid:
                    # 检查具体原因
                    has_result = RESULT_RE.search(out) is not None
                    has_search = SEARCH_RE.search(out) is not None
                    has_valid_dag = check_dag(out)
                    
                    logger.info(f"样本 {idx} 第一阶段输出格式不合规: "
                               f"has_search={has_search}, has_result={has_result}, has_valid_dag={has_valid_dag}")
                    
                    # 如果不合规是因为包含<result>标签，记录警告
                    if has_result:
                        logger.warning(f"样本 {idx} 第一阶段不应包含<result>标签，这可能导致重复的结果块")
                    
                    final_outputs[idx] = out  # 把第一阶段结果直接作为最终输出
                    active_mask[idx] = False  # 不再参与后续回合
           
            
            # 只对仍然活跃的样本进行检索
            live_indices = [i for i, (idx, active) in enumerate(zip(active_indices, active_mask)) 
                          if active_mask[idx]]
            if not live_indices:  # 如果没有活跃样本，直接进入下一轮
                continue
                
            live_outputs = [first_stage_outputs[i] for i in live_indices]
            live_mask = [True] * len(live_indices)
            
            logger.info(f"轮次 {turn+1}: 开始执行搜索，样本数: {len(live_outputs)}") # Log start of search execution
            search_start_time = time.time() # Record search start time
            # 执行搜索
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                live_outputs, 
                active_mask=live_mask
            )
            search_duration = time.time() - search_start_time
            logger.info(f"轮次 {turn+1}: 搜索执行完成，耗时 {search_duration:.2f} 秒。") # Log end of search execution
            
            # 更新活跃状态 (只处理仍然活跃的样本)
            for i, (live_idx, done, obs) in enumerate(zip(live_indices, dones, next_obs)):
                idx = active_indices[live_idx]  # 获取原始索引
                if done:
                    # 如果完成，加入最终结果
                    active_mask[idx] = False
                    final_outputs[idx] = live_outputs[i]
                else:
                    # 如果继续，加入观察结果
                    if obs:  # 只有在有结果时才添加
                        # 获取当前助手内容（应该只包含<think>和<search>部分）
                        current_assistant = self.convs[idx][-1]["content"]
                        
                        # 检查是否已有<result>标签，如果有则先移除，避免重复
                        if RESULT_RE.search(current_assistant):
                            logger.warning(f"第一阶段输出中发现<result>标签，将被移除并替换")
                            current_assistant = re.sub(RESULT_RE, '', current_assistant, flags=re.S | re.I)
                        
                        # 添加搜索结果
                        if obs:
                            # 确保 current_assistant 和 obs.strip() 之间有换行
                            self.convs[idx][-1]["content"] = current_assistant.rstrip() + "\n" + obs.strip()
                        else:
                            self.convs[idx][-1]["content"] = (
                                current_assistant.rstrip() + # rstrip()确保即使之前有空格也能正确添加换行
                                "\n<result>\n[无搜索结果或搜索失败]\n</result>\n\n"
                            )

                        # 更新缓存用来拼 final_outputs
                        first_stage_raw[idx] = self.convs[idx][-1]["content"]

            
            # 如果所有样本都完成，就退出循环
            if not any(active_mask):
                break
                
            # 第二阶段：为活跃样本生成最终答案
            active_indices = [i for i, active in enumerate(active_mask) if active]
            if not active_indices:
                break
                
            logger.info(f"轮次 {turn+1}: 开始第二阶段生成（最终答案），活跃样本数: {len(active_indices)}") 
            
            # 准备第二阶段输入
            # 修改：为每个活跃的会话添加一个新的assistant消息，内容为<answer>骨架
            active_convs = []
            for idx in active_indices:
                # 复制当前会话的消息列表(去掉最后一个assistant消息)
                # base_conv = self.convs[idx][:-1] if self.convs[idx] and self.convs[idx][-1]["role"] == "assistant" else self.convs[idx][:]
                base_conv = self.convs[idx][:]
                # 添加带有<answer>骨架的新assistant消息
                active_convs.append(base_conv + [{"role": "assistant", "content": "<answer>\n"}])
            
            second_stage_input_texts = [
                safe_chat_template(self.tokenizer, conv, add_generation_prompt=True) 
                for conv in active_convs
            ]
            
            # 使用_generate方法生成最终答案，修改stop_regex以确保在</answer>后停止
            final_stage_outputs = self._generate(
                prompts=second_stage_input_texts,
                max_tokens=max_tokens,
                stop_regex=r"</answer>\s*$",  # 修改正则表达式，确保在</answer>后停止
                temperature=0.6,
                top_p=0.9,
            )
            
            # 更新结果和会话
            for i, (idx, full_txt) in enumerate(zip(active_indices, final_stage_outputs)):
                # 只取第二阶段真正生成的 <answer>...</answer>
                ans_match = ANSWER_RE.search(full_txt)
                second_part = ans_match.group(0) if ans_match else full_txt

                # 更新会话缓存 - 修改：将原有的first_stage内容和新的answer合并
                if self.convs[idx][-1]["role"] == "assistant":
                    # 获取原有的第一阶段内容
                    first_stage_content = self.convs[idx][-1]["content"]
                    # 确保原有内容末尾没有多余的<answer>标签
                    if "<answer>" in first_stage_content:
                        first_stage_content = first_stage_content.split("<answer>")[0]
                    # 组合完整的助手回复
                    self.convs[idx][-1]["content"] = first_stage_content.rstrip() + "\n" + second_part.lstrip()
                else:
                    # 如果没有现有的assistant消息，直接添加
                    self.convs[idx].append({"role": "assistant", "content": second_part})

                # ⓐ 先拼出完整助手输出
                # 确保 first_stage_raw[idx] 和 second_part 之间有换行
                full_assistant = first_stage_raw[idx].rstrip() + "\n" + second_part.lstrip()

                # ⓑ 更新 final_outputs
                final_outputs[idx] = full_assistant

                # --- 添加调试日志 ---
                # if self.trainer.accelerator.is_main_process and idx == 0:
                #    logger.info("DEBUG final sample %d\n%s\n", idx, final_outputs[idx])
                # -----------------

                active_mask[idx] = False
        
        # 处理任何未完成的样本
        for i, active in enumerate(active_mask):
            if active:
                # 获取当前会话的最新状态
                final_outputs[i] = safe_chat_template(self.tokenizer, self.convs[i], add_generation_prompt=False)
        
        # 新增：分离提示部分和完成部分
        for i, output in enumerate(final_outputs):
            # 准备系统提示+用户问题作为prompt_parts (只包含输入部分)
            prompt_parts[i] = safe_chat_template(
                self.tokenizer, 
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": prompts[i]}],
                add_generation_prompt=True
            )
            # 将整个助手的输出作为completion_parts (包括<think>, <search>, <result>, <answer>)
            completion_parts[i] = output  # 确保包含完整的助手输出
                
        # 返回分离的提示部分和完成部分，而不是完整结果
        return prompt_parts, completion_parts

    def update_assistant_message(self, batch_idx, content, append=False):
        """更新或追加助手消息内容
        
        Args:
            batch_idx: 批次索引
            content: 要更新的内容
            append: 是否追加内容，而不是替换
            
        Returns:
            None
        """
        if 0 <= batch_idx < len(self.convs):
            # 检查最后一条消息是否是助手
            if self.convs[batch_idx] and self.convs[batch_idx][-1]["role"] == "assistant":
                if append:
                    # 追加内容
                    self.convs[batch_idx][-1]["content"] += content
                else:
                    # 替换内容
                    self.convs[batch_idx][-1]["content"] = content
            else:
                # 如果最后一条不是助手，添加新的助手消息
                self.convs[batch_idx].append({"role": "assistant", "content": content}) 

    def _generate(
        self,
        prompts: list[str],
        max_tokens: int,
        stop_regex: str = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ):
        """
        统一的采样封装：
        • trainer.use_vllm=True  → 复用GRPOTrainer里已经初始化好的vLLM
        • 否则                  → fallback到self.model.generate
        """
        logger = logging.getLogger(__name__)
        logger.info(f"开始执行_generate方法，prompts数量: {len(prompts)}, max_tokens: {max_tokens}, stop_regex (for transformers): {stop_regex}")
        
        # Trainer and vLLM related checks
        if self.trainer is None:
            raise ValueError("Trainer reference not set in GenerationManager.")

        use_vllm_from_config = getattr(self.trainer.args, "use_vllm", False)
        logger.info(f"[_generate] use_vllm_from_config: {use_vllm_from_config}")

        # 记录当前进程信息
        process_info = "未知进程"
        is_main = False
        if self.trainer and hasattr(self.trainer, 'accelerator'): # Ensure accelerator exists
            process_info = f"进程 {self.trainer.accelerator.process_index}/{self.trainer.accelerator.num_processes}"
            is_main = self.trainer.accelerator.is_main_process
            # logger.info(f"[_generate - {process_info}] {'主进程' if is_main else '非主进程'}") # Avoid excessive logging
        elif self.trainer and hasattr(self.trainer, 'is_main_process'): # Handle direct Accelerator case
             process_info = f"进程 {self.trainer.process_index}/{self.trainer.num_processes}"
             is_main = self.trainer.is_main_process
             # logger.info(f"[_generate - {process_info}] {'主进程' if is_main else '非主进程'}") # Avoid excessive logging
        # else: # Avoid excessive logging
            # logger.warning(f"[_generate] trainer或accelerator引用未设置，无法确定详细进程信息")


        if use_vllm_from_config:
            logger.info(f"[_generate - {process_info}] 使用 vLLM 进行生成 (vLLM client: {self.trainer.vllm_client is not None})")
            if self.trainer.vllm_client is None:
                logger.error(f"[_generate - {process_info}] vLLM is enabled but vllm_client is not initialized in trainer. Falling back to transformers.generate.")
                # Fallback logic is already below
            else:
                try:
                    from vllm import SamplingParams
                    # LoRARequest is not directly used here if adapter is pre-loaded by GRPOTrainer

                    # stop_sequences are derived from vllm_guided_decoding_regex in GRPOTrainer's _init_vllm_client
                    # and stored in self.trainer.stop_sequences.
                    sampling_params = SamplingParams(
                        temperature=temperature if temperature > 0 else 1.0, 
                        top_p=top_p if temperature > 0 else 1.0, 
                        top_k=-1, 
                        max_tokens=max_tokens,
                        stop=self.trainer.stop_sequences, # Use stop_sequences from GRPOTrainer
                        include_stop_str_in_output=True, 
                    )
                    if is_main: # Log sampling params only from main process to reduce noise
                        logger.info(f"[_generate - {process_info} - vLLM] SamplingParams: {sampling_params}")
                        logger.info(f"[_generate - {process_info} - vLLM] Using stop_sequences from trainer: {self.trainer.stop_sequences}")

                    # GRPOTrainer pre-loads the LoRA adapter to the vLLM server via _move_model_to_vllm.
                    # The generate call implicitly uses the active LoRA.
                    if hasattr(self.trainer, 'lora_adapter_name') and self.trainer.lora_adapter_name and is_main:
                        logger.info(f"[_generate - {process_info} - vLLM] Active LoRA expected on server: {self.trainer.lora_adapter_name}")
                    
                    request_ids = [f"grpo-gen-{self.trainer.state.global_step if self.trainer.state else 'unk'}-{i}" for i in range(len(prompts))] 

                    logger.info(f"[_generate - {process_info} - vLLM] Calling vllm_client.generate with {len(prompts)} prompts.")
                    gen_start_time = time.time()
                    
                    # The prompts here are already the full input text for the model
                    outputs = self.trainer.vllm_client.generate(
                        prompts,
                        sampling_params,
                        request_ids,
                    )
                    gen_duration = time.time() - gen_start_time
                    logger.info(f"[_generate - {process_info} - vLLM] vLLM generation completed in {gen_duration:.2f}s. Received {len(outputs)} outputs.")

                    generated_texts_with_prompt = []
                    for i, req_output in enumerate(outputs):
                        current_prompt_for_this_req = prompts[i] 
                        if req_output.outputs:
                            generated_suffix = req_output.outputs[0].text
                            generated_texts_with_prompt.append(current_prompt_for_this_req + generated_suffix)
                            if i == 0 and is_main: 
                                logger.info(f"[_generate - {process_info} - vLLM] Sample 0 vLLM output (suffix): '{generated_suffix[:200]}...'")
                                logger.info(f"[_generate - {process_info} - vLLM] Sample 0 vLLM full (prompt+suffix): '{(current_prompt_for_this_req + generated_suffix)[:300]}...'")
                        else:
                            logger.warning(f"[_generate - {process_info} - vLLM] Request {req_output.request_id} produced no output. Returning prompt only.")
                            generated_texts_with_prompt.append(current_prompt_for_this_req) 
                    
                    if len(generated_texts_with_prompt) != len(prompts):
                        logger.error(f"[_generate - {process_info} - vLLM] Mismatch in number of generated texts ({len(generated_texts_with_prompt)}) and prompts ({len(prompts)}).")
                        raise RuntimeError("vLLM generation output count mismatch.")

                    return generated_texts_with_prompt

                except Exception as e:
                    logger.error(f"[_generate - {process_info}] Error during vLLM generation: {e}. Falling back to transformers.generate.")
                    # Fallback logic is below

        # ---------- 使用transformers.generate ----------
        logger.info(f"[_generate - {process_info}] 使用transformers.generate进行生成")
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        
        target_device = None
        if self.trainer and hasattr(self.trainer, 'accelerator'):
            target_device = self.trainer.accelerator.device
        elif self.trainer and hasattr(self.trainer, 'device'): # Direct Accelerator case
            target_device = self.trainer.device
        elif hasattr(self.model, 'device'):
            target_device = self.model.device
        else:
            logger.warning("[_generate] 无法确定目标设备，将使用默认设备")

        if target_device:
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            logger.info(f"[_generate - {process_info}] 输入已移动到设备: {target_device}")
        else:
            logger.info(f"[_generate - {process_info}] 输入保持在默认设备")
            
        # 为transformers创建stopping criteria
        stopping_criteria = None
        if stop_regex:
            from transformers import StoppingCriteria, StoppingCriteriaList
            
            class RegexStoppingCriteria(StoppingCriteria):
                def __init__(self, tokenizer, regex_pattern, prompt_len):
                    import re
                    self.tokenizer = tokenizer
                    self.pattern = re.compile(regex_pattern)
                    self.prompt_len = prompt_len  # 初始prompt长度
                    self.stopped_sequences = set()
                    
                def __call__(self, input_ids, scores, **kwargs):
                    for i, sequence in enumerate(input_ids):
                        if i in self.stopped_sequences:
                            continue
                        # 只解码新生成的部分，避免匹配到prompt中的示例
                        new_text = self.tokenizer.decode(sequence[self.prompt_len:])
                        if self.pattern.search(new_text):
                            self.stopped_sequences.add(i)
                    return len(self.stopped_sequences) == len(input_ids)
            
            # 获取prompt的token长度
            prompt_lens = inputs["input_ids"].size(1)
            
            stopping_criteria = StoppingCriteriaList([
                RegexStoppingCriteria(self.tokenizer, stop_regex, prompt_len=prompt_lens)
            ])
            logger.info(f"[_generate - {process_info}] 已创建正则表达式停止条件: {stop_regex}")
        
        logger.info(f"[_generate - {process_info}] 开始生成 (model: {type(self.model).__name__})...")
        gen_start_time = time.time()
        
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
            )
            # 如果输入就是输出（例如，max_new_tokens=0 或由于某些原因没有生成新标记），
            # gen_ids 可能只包含输入 IDs。我们需要确保只获取新生成的 tokens。
            input_ids_length = inputs["input_ids"].size(1)
            if gen_ids.size(1) > input_ids_length:
                 gen_ids = gen_ids[:, input_ids_length:]
            else: # 没有生成新的 token
                 gen_ids = torch.empty((gen_ids.size(0),0), dtype=torch.long, device=gen_ids.device) # 返回空 tensor
            
        gen_duration = time.time() - gen_start_time
        logger.info(f"[_generate - {process_info}] 生成完成，耗时: {gen_duration:.2f}秒, 生成了 {gen_ids.shape[1]} 个新 tokens")
        
        # 解码生成的ID
        decode_start_time = time.time()
        result = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        decode_duration = time.time() - decode_start_time
        
        logger.info(f"[_generate - {process_info}] 解码完成，耗时: {decode_duration:.2f}秒，解码了 {len(result)} 个文本")
        
        if result and is_main: # Log the first result for debugging from main process
            logger.info(f"[_generate - {process_info}] (transformers.generate) Sample 0 HF output: '{result[0][:300]}...'")
            
        return result

def test_proxy_connection():
    """测试代理连接是否正常工作
    
    这个函数会通过代理发起一个请求，并打印返回的IP地址
    如果成功，应该显示代理的IP而不是本地IP
    
    用法：
    ```
    if __name__ == "__main__":
        test_proxy_connection()
    ```
    """
    import requests
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # 使用PROXY_DICT发送请求
        logger.info(f"测试代理配置: {PROXY_DICT}")
        response = requests.get("https://api.ipify.org", proxies=PROXY_DICT, timeout=10)
        proxy_ip = response.text.strip()
        logger.info(f"通过代理获取的IP: {proxy_ip}")
        
        # 不使用代理，获取真实IP进行对比
        direct_ip = None
        try:
            direct_response = requests.get("https://api.ipify.org", timeout=10)
            direct_ip = direct_response.text.strip()
            logger.info(f"直接连接获取的IP: {direct_ip}")
        except Exception as e:
            logger.warning(f"无法直接连接获取IP: {str(e)}")
        
        # 测试DuckDuckGoSearch
        try:
            from lagent.actions.web_browser import DuckDuckGoSearch
            logger.info("测试DuckDuckGoSearch代理配置...")
            
            # 创建搜索实例，使用proxies参数
            search = DuckDuckGoSearch(proxies=PROXY_DICT, timeout=10)
            
            # 执行搜索
            results = search.search("test query")
            if results:
                logger.info(f"DuckDuckGoSearch搜索成功，返回 {len(results)} 个结果。")
                # 显示第一个结果
                if len(results) > 0:
                    logger.info(f"第一个结果: {next(iter(results.values()))}")
            else:
                logger.warning("DuckDuckGoSearch搜索返回空结果")
        except Exception as e:
            logger.error(f"DuckDuckGoSearch测试失败: {str(e)}")
        
        # 检查是否使用了代理
        if direct_ip and proxy_ip != direct_ip:
            logger.info("代理配置正常工作！")
            return True
        elif direct_ip is None:
            logger.info("代理配置似乎可以工作（无法直接连接进行对比）")
            return True
        else:
            logger.warning("代理可能未生效：通过代理和直接连接获取的IP地址相同")
            return False
    except Exception as e:
        logger.error(f"测试代理时出错: {str(e)}")
        return False

def test_search_tools():
    """测试SearchTools类的搜索功能"""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("初始化SearchTools...")
        search_tools = SearchTools()
        
        # 测试网页搜索
        logger.info("测试网页搜索...")
        web_results = search_tools.search_web("Python programming")
        logger.info(f"网页搜索结果: {web_results[:200]}...")
        
        # 测试学术搜索
        logger.info("测试学术搜索...")
        academy_results = search_tools.search_academy("transformer architecture")
        logger.info(f"学术搜索结果: {academy_results[:200]}...")
        
        logger.info("所有测试完成！")
        return True
    except Exception as e:
        logger.error(f"测试搜索工具时出错: {str(e)}")
        return False

# 如果直接运行此文件，执行测试
if __name__ == "__main__":
    print("=== 测试代理连接 ===")
    proxy_ok = test_proxy_connection()
    
    if proxy_ok:
        print("\n=== 测试搜索工具 ===")
        test_search_tools() 