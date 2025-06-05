import os

import json
import re
import math
import random
import torch
import logging
import argparse
import openai
import wandb
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import html
import time
from functools import lru_cache  # 添加lru_cache导入

# 导入搜索工具管理器
from search_manager import SearchTools, DAGParser, GenerationManager
from search_manager import THINK_RE, SEARCH_RE, RESULT_RE, ANSWER_RE, NODE_RE, EDGE_RE
from search_manager import check_r1, check_dag, safe_chat_template

# 导入trl库
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from accelerate.utils import gather, gather_object, broadcast_object_list
from trl.extras.profiling import profiling_context

from torch import nn
from datasets import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, StoppingCriteria, StoppingCriteriaList

# 在导入部分添加PEFT相关库
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

# 导入Accelerate
from accelerate import Accelerator

# 设置日志
log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, 'train_rlsf.log')

# 确保日志目录存在
os.makedirs(log_dir, exist_ok=True)
# os.environ["VLLM_DISABLE_NCCL_COMM"] = "1"  
# logging.warning("VLLM_DISABLE_NCCL_COMM = %s", os.getenv("VLLM_DISABLE_NCCL_COMM"))

# 创建全局logger
logger = logging.getLogger(__name__)

def get_chat_special_tokens(tok):
    """
    返回 (im_start, im_end)；如果模型没有定义就退回默认占位符。
    """
    m = tok.special_tokens_map
    im_start = m.get("im_start", "<|im_start|>")
    im_end   = m.get("im_end",   "<|im_end|>")
    return im_start, im_end

SYSTEM_PROMPT_CN = """
你是一名检索规划代理。对任何用户问题，你必须严格输出 4 段标签，顺序固定，不得缺段、不得串行：

<think> … </think>
<search> … </search>
<result> … </result>
<answer> … </answer>

### 规则
1. **<think>** 标签内推理需要：
   - 首先拆分用户问题的关键词/概念
   - 分析每个关键词最适合用哪种搜索工具
   - 确定搜索顺序和节点间的依赖关系
   - 字数不限，可跨多段
2. **<search>** 必须包含 "Nodes:" 与 "Edges:" 两节，格式严格如下（大小写敏感）  
   - Nodes 节按 标签: 查询描述 (工具) 一行一个。  
   - Edges 节按 源标签 -> 目标标签，多条用分号隔开。  
3. 限制  
   - 节点 ≤ 8；每个查询描述 ≤ 10 词；  
   - 工具 3 选 1 且对每节点唯一：  
     • **General** = 网页搜索（Google）  
     • **News** = 新闻搜索（GNews）  
     • **Academy** = 论文搜索（ArXiv，需英文关键词）  
   - 不得出现自环或环路；Edges 只允许已声明节点。  
4. **<result>** 该标签由系统在 **第二阶段** 自动填充。**第一阶段禁止写出此标签**
5. **<answer>** 基于结果给最终答案，语言自选但需完整准确。  
6. 回复中除以上标签外，禁止出现其他自定义标签或 Markdown。  

### 示例

<think>
问题：美国关税政策对 401(k) 退休账户和物价有什么影响？

关键词与工具  
- "US tariffs 2024"   → 最新政策 → News  
- "tariffs impact 401k returns and CPI" → 市场/物价解读 → General  

依赖：先拿到最新政策，再看市场和物价分析。
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
【系统自动填充】
</result>

<answer>
短期内股市波动不应过度反应，长期关税持续可能降低企业利润率；物价可能上涨但需时间反映，企业转嫁成本有风险，消费者可关注多品牌选择。
</answer>
"""

# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (torch.Tensor):
            Input tensor of shape (N,).

    Returns:
        torch.Tensor:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

def encode_chat_messages(tokenizer, messages, add_generation_prompt=True):
    """通用的消息编码函数，确保正确应用聊天模板
    
    Args:
        tokenizer: 分词器
        messages: 消息列表，格式为[{"role": "system", "content": ...}, {"role": "user", "content": ...}, ...]
        add_generation_prompt: 是否添加生成提示
        
    Returns:
        应用了聊天模板的文本
    """
    # 使用共享的safe_chat_template函数
    return safe_chat_template(tokenizer, messages, add_generation_prompt)

class ExternalRewardModel:
    """
    连续评分奖励模型，根据问题类型返回0.00-1.00之间的浮点数评分。
    """
    def __init__(self, 
                 api_key, 
                 model="gpt-4.1-mini-2025-04-14",
                 score_mode="continuous",  # 新增参数，支持"continuous"或"discrete"
                 partial_credit: float = 0.25,
                 ):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.openai-proxy.com/v1",
        )
        self.model = model
        self.score_mode = score_mode
        self.partial_credit = max(0.0, min(partial_credit, 1.0))

    # --------------------------- prompt ----------------------------
    # 离散评分提示词
    _DISCRETE_PROMPT = (
        "你是一个严格的AI评分助手，根据问题和参考答案来评价待评估的回答质量。"
        "你只能输出以下三个词之一："
        "\"CORRECT\" (完全正确)，"
        "\"PARTIAL\" (部分正确)，"
        "\"WRONG\" (错误)。\n\n"
        "判断标准：\n"
        "- CORRECT：回答与参考答案在事实、数据、结论和核心观点上高度一致，覆盖了所有关键信息。即使表述方式不同，只要核心内容准确完整，也应判定为CORRECT。\n"
        "- PARTIAL：回答包含部分正确信息，但存在轻度遗漏、轻度错误或未完全解答问题。如果回答包含了大部分核心信息但缺少某些细节，应判定为PARTIAL。\n"
        "- WRONG：回答包含严重错误、与参考答案矛盾、完全偏离问题或信息严重不足。如果回答仅包含表面性内容而缺失关键信息，也应判定为WRONG。\n\n"
        "在判断时，请特别注意以下几点：\n"
        "1. 重点评估信息的准确性和完整性，而非表达方式\n"
        "2. 即使回答风格或组织方式与参考答案不同，只要核心内容匹配，也可判定为CORRECT\n"
        "3. 对于有多个要点的问题，应确保所有重要要点都被准确回答\n\n"
        "请仔细对比问题、参考答案和待评回答，确保评判公正准确。禁止输出其他内容。"
    )
    
    # 连续评分提示词 - 通用版本
    _CONTINUOUS_PROMPT = (
        "你是一个严格的AI评分助手，根据问题和参考答案来评价待评估的回答质量。\n"
        "请阅读问题、参考答案和待评回答，返回0~1之间的分数（越接近1表示越正确）。\n"
        "只输出一个数值，保留两位小数，禁止输出任何其他文字或符号。\n\n"
        "评分标准：\n"
        "- 1.00分：回答与参考答案在事实、数据、结论和核心观点上高度一致，覆盖了所有关键信息\n"
        "- 0.75分：回答包含大部分正确信息，但有轻微遗漏或细节不够完整\n"
        "- 0.50分：回答包含一半正确信息，同时有一半错误或缺失\n"
        "- 0.25分：回答仅包含少量正确信息，大部分内容错误或缺失\n"
        "- 0.00分：回答完全错误或与问题无关\n\n"
        "请基于信息的准确性和完整性进行评分，而非表达方式。即使表述不同，只要核心内容正确也可得高分。"
    )
    
    # 连续评分提示词 - 多选题版本
    _MULTI_PROMPT = (
        "你是一个严格的AI评分助手，根据问题和参考答案来评价待评估的多选题回答质量。\n"
        "请先判断每个选项的正确与否，然后计算微平均F1分数（正确选中数/总选中数与正确选中数/参考答案选项数的调和平均值）。\n"
        "最后输出一个0~1之间的分数，保留两位小数，禁止输出任何其他文字或符号。\n\n"
        "评分示例：\n"
        "- 如果参考答案是A,B,C，待评回答选了A,B,C,D，则精确率=3/4，召回率=3/3, F1=(2*3/4*3/3)/(3/4+3/3)=0.86\n"
        "- 如果参考答案是A,B,C，待评回答只选了A,B，则精确率=2/2，召回率=2/3，F1=(2*2/2*2/3)/(2/2+2/3)=0.80\n"
        "- 如果参考答案是A,B，待评回答选了C,D，则F1=0.00\n"
    )
    
    def _judge(self, question: str, answer: str, reference: str, question_type: str = "fact") -> float:
        """
        根据问题类型和评分模式判断答案质量
        
        Args:
            question: 原始问题
            answer: 待评估的回答
            reference: 参考答案
            question_type: 问题类型，可以是'fact'/'cause'/'multi'
            
        Returns:
            float: 0到1之间的分数
        """
        # 提取<answer>部分
        answer_match = ANSWER_RE.search(answer)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # 查找</result>标签的位置
            result_end = re.search(r'</result>', answer, flags=re.I)
            if result_end:
                # 只取</result>标签后面的内容作为答案
                answer = answer[result_end.end():].strip()
            else:
                # 如果连</result>标签都没有，则移除所有标签内容
                answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.S | re.I)
                answer = re.sub(r'<search>.*?</search>', '', answer, flags=re.S | re.I)
                answer = re.sub(r'<result>.*?</result>', '', answer, flags=re.S | re.I)
                answer = answer.strip()
        
        # 根据score_mode选择不同的系统提示
        if self.score_mode == "discrete":
            system_prompt = self._DISCRETE_PROMPT
        elif question_type.lower() == "multi":
            system_prompt = self._MULTI_PROMPT
        else:
            system_prompt = self._CONTINUOUS_PROMPT
            
        user_msg = (
            f"【问题】{question}\n\n"
            f"【参考答案】{reference}\n\n"
            f"【待评回答】{answer}\n\n"
            "请给出评分："
        )
        
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=4,  # 简短输出就够了
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg}
                ]
            )
            
            response_text = rsp.choices[0].message.content.strip().upper()
            
            # 处理返回结果
            if self.score_mode == "discrete":
                # 离散评分模式
                if response_text == "CORRECT":
                    score = 1.0
                elif response_text == "PARTIAL":
                    score = self.partial_credit
                else:
                    score = 0.0
            else:
                # 连续评分模式 - 尝试将输出解析为浮点数
                try:
                    # 移除可能的非数字字符
                    score_text = re.sub(r'[^\d.]', '', response_text)
                    score = float(score_text)
                    # 确保分数在0-1范围内
                    score = max(0.0, min(score, 1.0))
                except (ValueError, TypeError):
                    # 如果无法解析为浮点数，则返回0
                    logger.warning(f"无法将 '{response_text}' 解析为浮点数，默认给0分")
                    score = 0.0
            
            # 打印关键信息
            logger.info(
                "[OpenAI] 分数=%s  tokens_prompt=%d  tokens_resp=%d",
                score,
                rsp.usage.prompt_tokens,
                rsp.usage.completion_tokens,
            )
            return score
            
        except Exception as e:
            logger.error(f"调用评分API时出错: {str(e)}")
            return 0.0  # 出错返回最低分
    
    @lru_cache(maxsize=100_000)
    def _judge_cached(self, question: str, answer: str, reference: str, question_type: str) -> float:
        """缓存版本的判断方法，提高性能
        
        Args:
            question: 原始问题
            answer: 待评估的回答
            reference: 参考答案
            question_type: 问题类型，可以是'fact'/'cause'/'multi'
            
        Returns:
            float: 0到1之间的分数
        """
        return self._judge(question, answer, reference, question_type)
    
    def semantic_score(self,
                       question: str,
                       answer: str,
                       reference: str,
                       question_type: str = "fact") -> float:
        """
        返回0-1之间的连续分数。
        
        Args:
            question: 原始问题
            answer: 待评估的回答
            reference: 参考答案
            question_type: 问题类型，可以是'fact'/'cause'/'multi'
            
        Returns:
            float: 0到1之间的分数
        """
        try:
            # 使用缓存版本的判断方法
            score = self._judge_cached(question, answer, reference, question_type)
            # 确保分数在0-1范围内
            return max(0.0, min(score, 1.0))
        except Exception as e:
            logger.error(f"评分过程中出错: {str(e)}")
            return 0.0  # 出错返回最低分

def load_dataset(file_path: str) -> List[Dict]:
    """加载并处理数据集，支持JSON和JSONL格式
    
    Args:
        file_path: 数据集文件路径，可以是.json或.jsonl格式
        
    Returns:
        数据列表
    """
    if file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data.append(json.loads(line))
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]  # 如果JSON文件包含单个对象，转换为列表
    
    return data

class SearchAwareGRPOTrainer(GRPOTrainer):
    """支持搜索功能的GRPO训练器"""
    
    def __init__(self, *args, generation_manager=None, **kwargs):
        """初始化训练器
        
        Args:
            generation_manager: 生成管理器实例, 必须提供。
            其他参数同GRPOTrainer
        """
        if generation_manager is None:
            raise ValueError("SearchAwareGRPOTrainer 需要一个 generation_manager 实例")
            
        super().__init__(*args, **kwargs)  # Accelerator 会在这里准备模型
        self.generation_manager = generation_manager
        # 关键：将 Accelerator 处理过的模型同步回 GenerationManager
        self.generation_manager.model = self.model
        self.verbose_logging = True
        # self.step_counter = 0
        self.use_wandb = getattr(self.args, "use_wandb", False)
    
    def _generate_and_score_completions(self, inputs):
        """
        重写GRPOTrainer的_generate_and_score_completions方法
        这是模型生成和评分的核心方法
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # 提取提示
        prompts = [x["prompt"] for x in inputs]
        
        # 提取sample_id用于reward_fn，确保在用dataloader.shuffle()时能匹配正确的参考答案
        sample_ids = [x.get("sample_id", None) for x in inputs]
        
        # 确保 trainer 引用传递给 generation_manager (如果需要内部调用 accelerator.generate)
        if hasattr(self.generation_manager, 'set_trainer_reference'):
             self.generation_manager.set_trainer_reference(self)
        
        # 记录vLLM状态信息
        if self.accelerator.is_main_process:
            self.verbose_logging = True  # 确保详细日志记录
            logger.info(f"[_generate_and_score_completions] 开始生成和评分，模式: {mode}, 提示数量: {len(prompts)}")
            logger.info(f"[_generate_and_score_completions] use_vllm设置为: {getattr(self, 'use_vllm', False)}")
            if hasattr(self, 'vllm_client') and getattr(self, 'use_vllm', False):
                logger.info(f"[_generate_and_score_completions] vllm_client存在: {self.vllm_client is not None}")
            else:
                logger.warning(f"[_generate_and_score_completions] vllm_client不存在或use_vllm=False")
        
        # 在真正调用self.generation_manager.generate_with_search之前：
        if getattr(self, "use_vllm", False):
            # 把最新LoRA权重push给vLLM（与原GRPOTrainer完全一致）
            try:
                logger.info(f"[_generate_and_score_completions] 开始将LoRA权重推送到vLLM...")
                push_start_time = time.time()
                
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
                
                push_duration = time.time() - push_start_time
                logger.info(f"[_generate_and_score_completions] LoRA权重推送完成，耗时: {push_duration:.2f}秒")
            except Exception as e:
                logger.error(f"[_generate_and_score_completions] 推送LoRA权重到vLLM时出错: {str(e)}")
        
        # 使用generation_manager生成带搜索的提示和完成部分
        try:
            logger.info(f"[_generate_and_score_completions] 开始调用generation_manager.generate_with_search...")
            gen_start_time = time.time()
            
            # 修改：接收分离的提示部分和完成部分，而不是完整响应
            prompts_text, completions_text = self.generation_manager.generate_with_search(
                prompts,
                max_completion_length=self.max_completion_length,
            )
            
            gen_duration = time.time() - gen_start_time
            logger.info(f"[_generate_and_score_completions] generate_with_search完成，耗时: {gen_duration:.2f}秒")
            
            if len(prompts_text) != len(prompts) or len(completions_text) != len(prompts):
                logger.warning(f"[_generate_and_score_completions] 返回的提示/完成数量不匹配: prompts={len(prompts)}, prompts_text={len(prompts_text)}, completions_text={len(completions_text)}")
        except Exception as e:
            logger.error(f"[_generate_and_score_completions] 生成过程中发生异常: {str(e)}")
            raise
        
        # 处理提示文本部分（包含系统提示和助手骨架）
        logger.info(f"[_generate_and_score_completions] 开始处理提示文本...")
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = {k: v.to(self.accelerator.device) for k, v in prompt_inputs.items()}

        if self.max_prompt_length is not None:
            prompt_inputs = {k: v[:, -self.max_prompt_length:] for k, v in prompt_inputs.items()}
        
        # 编码生成的完成内容（仅包含<result>和<answer>部分）
        logger.info(f"[_generate_and_score_completions] 开始处理完成文本...")
        completion_inputs = self.processing_class(
            text=completions_text, 
            return_tensors="pt", 
            padding=True, 
            padding_side="right", 
            add_special_tokens=False
        )
        completion_inputs = {k: v.to(self.accelerator.device) for k, v in completion_inputs.items()}
        
        # 检查是否有格式错误，提供日志
        format_check_start = time.time()
        logger.info(f"[_generate_and_score_completions] 开始格式检查...")
        
        format_errors = 0
        for i, comp in enumerate(completions_text):
            # r1_ok = check_r1(prompts_text[i] + comp)  # 合并检查完整输出格式
            # dag_ok = check_dag(prompts_text[i] + comp)  # 合并检查完整DAG格式
            r1_ok = check_r1(comp)  # 合并检查完整输出格式
            dag_ok = check_dag(comp)  # 合并检查完整DAG格式
            if not (r1_ok and dag_ok):
                format_errors += 1
                if self.verbose_logging and self.accelerator.is_main_process:
                    logger.warning(f"Sample #{i} format check failed: r1_ok={r1_ok}, dag_ok={dag_ok}")
        
        format_check_duration = time.time() - format_check_start
        logger.info(f"[_generate_and_score_completions] 格式检查完成，耗时: {format_check_duration:.2f}秒，发现 {format_errors} 个格式错误")
                    
        # 屏蔽<result>部分，在attention_mask中将这部分设为0
        mask_result_start = time.time()
        logger.info(f"[_generate_and_score_completions] 开始屏蔽<result>部分...")
        
        for b, comp_txt in enumerate(completions_text):
            result_match = RESULT_RE.search(comp_txt)
            if result_match:
                # 找到了<result>块
                char_start, char_end = result_match.span()
                
                # 将字符级别的位置转换为token位置
                result_prefix_tokens = len(self.processing_class(
                    comp_txt[:char_start], 
                    add_special_tokens=False
                )["input_ids"])
                
                result_tokens = len(self.processing_class(
                    comp_txt[char_start:char_end], 
                    add_special_tokens=False
                )["input_ids"])
                
                # 直接在completion_mask中屏蔽<result>部分
                result_start_pos = result_prefix_tokens
                result_end_pos = result_start_pos + result_tokens
                
                # 确保索引在有效范围内
                result_end_pos = min(result_end_pos, completion_inputs["attention_mask"].size(1))
                
                if result_start_pos < completion_inputs["attention_mask"].size(1):
                    completion_inputs["attention_mask"][b, result_start_pos:result_end_pos] = 0
                
                if self.verbose_logging and self.accelerator.is_main_process and b == 0:
                    logger.info(f"已屏蔽completion_mask中的result部分 tokens {result_start_pos}-{result_end_pos} for sample {b}")
        
        mask_result_duration = time.time() - mask_result_start
        logger.info(f"[_generate_and_score_completions] 屏蔽<result>部分完成，耗时: {mask_result_duration:.2f}秒")
        
        # 使用处理过的labels进行后续计算
        # 处理生成的完成内容以供训练使用
        # 创建一个带有<result>标签的副本用于reward计算
        # 注意：这里不需要额外strip，因为现在的completions_text已经只包含<result>和<answer>部分
        stripped_completions = completions_text
        
        # 计算奖励
        rewards_start = time.time()
        logger.info(f"[_generate_and_score_completions] 开始计算奖励...")
        
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                    # 对于模型类型的奖励函数，需要进行特殊处理
                    if is_conversational(inputs[0]):
                        # 构建完整对话，但用stripped_completions作为模型生成部分
                        messages = [{"messages": p + c} for p, c in zip(prompts_text, stripped_completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        # 简单连接prompts_text和stripped_completions
                        texts = [p + c for p, c in zip(prompts_text, stripped_completions)]
                    
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = {k: v.to(self.accelerator.device) for k, v in reward_inputs.items()}
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    # 对于自定义奖励函数
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    # 传递sample_ids给reward_func
                    reward_kwargs["sample_id"] = sample_ids
                    # 使用分离后的prompts_text和stripped_completions计算奖励
                    output_reward_func = reward_func(prompts=prompts_text, completions=stripped_completions, **reward_kwargs)
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        rewards_duration = time.time() - rewards_start
        logger.info(f"[_generate_and_score_completions] 奖励计算完成，耗时: {rewards_duration:.2f}秒")
        
        # 从这里开始可以大致保留原始GRPOTrainer中的代码
        # 汇集奖励，计算优势等
        
        # 收集奖励 - 这是GRPOTrainer中的关键部分
        gather_start = time.time()
        logger.info(f"[_generate_and_score_completions] 开始收集奖励...")
        
        rewards_per_func = gather(rewards_per_func)
        
        gather_duration = time.time() - gather_start
        logger.info(f"[_generate_and_score_completions] 奖励收集完成，耗时: {gather_duration:.2f}秒")
        
        # 应用权重并求和
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        
        # 计算分组奖励
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # 计算优势
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)
            
        # 切片保留本地部分数据
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        
        # 记录指标
        if mode == "train":
            attention_mask = torch.cat([prompt_inputs["attention_mask"], completion_inputs["attention_mask"]], dim=1)
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        
        # 记录完成内容长度相关指标
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_inputs["attention_mask"].sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())
        
        # 记录终止序列的相关指标
        is_eos = completion_inputs["input_ids"] == self.processing_class.eos_token_id
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        
        # 处理空终止序列的边缘情况
        if len(term_completion_mask) == 0:
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())
        
        # 计算每个奖励函数的平均奖励
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        
        # 记录提示和完成文本 - 使用我们分离后的版本
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        
        # 计算旧的logps（如果需要）
        logps_start = time.time()
        
        if self.num_iterations > 1:
            logger.info(f"[_generate_and_score_completions] 开始计算旧的logps...")
            # 创建完整的输入序列
            input_ids = torch.cat([prompt_inputs["input_ids"], completion_inputs["input_ids"]], dim=1)
            attention_mask = torch.cat([prompt_inputs["attention_mask"], completion_inputs["attention_mask"]], dim=1)
            logits_to_keep = completion_inputs["input_ids"].size(1)
            
            with torch.no_grad():
                old_per_token_logps = self._get_per_token_logps(
                    self.model, input_ids, attention_mask, logits_to_keep, self.args.per_device_train_batch_size
                )
                
            logps_duration = time.time() - logps_start
            logger.info(f"[_generate_and_score_completions] 计算旧的logps完成，耗时: {logps_duration:.2f}秒")
        else:
            old_per_token_logps = None
            logger.info(f"[_generate_and_score_completions] 跳过计算旧的logps (num_iterations=1)")
        
        total_duration = time.time() - gen_start_time
        logger.info(f"[_generate_and_score_completions] 完成生成和评分过程，总耗时: {total_duration:.2f}秒")
        
        # 返回处理后的输入，不再包含labels
        return {
            "prompt_ids": prompt_inputs["input_ids"],
            "prompt_mask": prompt_inputs["attention_mask"],
            "completion_ids": completion_inputs["input_ids"],
            "completion_mask": completion_inputs["attention_mask"],
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }

def train(args):
    """训练函数"""
    # 在trainer初始化前，需要使用临时accelerator处理日志配置
    temp_accelerator = Accelerator()
    
    # 使用临时accelerator判断主进程
    if temp_accelerator.is_main_process:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  
                logging.FileHandler(log_file)
            ]
        )
        logger.info(f"日志级别设置为: {log_level}")
        logger.info(f"日志文件保存在: {log_file}")
        
        # 打印训练参数
        logger.info("=== 训练参数 ===")
        for arg_name, arg_value in sorted(vars(args).items()):
            logger.info(f"  {arg_name}: {arg_value}")
        logger.info("===============")
        
        # vLLM参数日志
        if args.use_vllm:
            logger.info(f"=== vLLM配置 ===")
            logger.info(f"  use_vllm: {args.use_vllm}")
            logger.info(f"  vllm_server_host: {args.vllm_server_host}")
            logger.info(f"  vllm_server_port: {args.vllm_server_port}")
            # logger.info(f"  vllm_mode: {args.vllm_mode}")
            logger.info("===============")

            # 检查vLLM服务器是否可访问
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                result = s.connect_ex((args.vllm_server_host, args.vllm_server_port))
                if result == 0:
                    logger.info(f"vLLM服务器在 {args.vllm_server_host}:{args.vllm_server_port} 可访问。")
                else:
                    logger.error(f"无法连接到vLLM服务器 {args.vllm_server_host}:{args.vllm_server_port}。确保服务器正在运行。")
                s.close()
            except Exception as e:
                logger.error(f"检查vLLM服务器连接时出错: {str(e)}")
        
        # 初始化wandb，只在主进程上进行
        if args.use_wandb:
            wandb.login(key=args.wandb_api_key)
            # 添加默认的项目名和运行名称
            project_name = getattr(args, "wandb_project", "rlsf-training")
            run_name = getattr(args, "wandb_run_name", f"rlsf-{args.model_name.split('/')[-1]}")
            wandb.init(
                project=project_name,
                name=run_name,
                config=args
            )
            logger.info("成功初始化wandb")
    else:
        # 非主进程禁用日志
        logging.basicConfig(level=logging.ERROR)
    
    if temp_accelerator.is_main_process:
        logger.info("正在加载tokenizer进行聊天模板自检...")
        try:
            temp_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            test_messages = [
                {"role": "system", "content": "你是一个AI助手。"},
                {"role": "user", "content": "你好"}
            ]
            test_result = encode_chat_messages(temp_tokenizer, test_messages)
            
            if temp_tokenizer.bos_token in test_result:
                logger.info(f"聊天模板自检通过: {test_result[:50]}...")
            else:
                logger.warning(f"聊天模板自检警告: 输出中缺少bos_token标记。将使用fallback。")
                
            if not temp_tokenizer.chat_template:
                logger.warning("警告: 模型没有chat_template，将使用fallback。")
                
            if temp_tokenizer.pad_token_id is None:
                temp_tokenizer.pad_token = temp_tokenizer.eos_token
                logger.info(f"Tokenizer 未定义 pad_token，已设置为 eos_token: {temp_tokenizer.eos_token}")
                
            # 测试聊天模板
            logger.info("测试聊天模板...")
            try:
                # 构建测试消息
                test_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_CN},
                    {"role": "user", "content": "这是一个测试问题"},
                    {"role": "assistant", "content": "<think>\n"} 
                ]
                
                # 应用聊天模板
                templated_text = temp_tokenizer.apply_chat_template(
                    test_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # 记录模板应用结果
                logger.info(f"聊天模板测试成功，输出前100个字符: {templated_text[:100]}...")
                
                # 检查是否包含关键标记（使用bos_token和eos_token）
                if temp_tokenizer.bos_token in templated_text:
                    logger.info("聊天模板包含正确的bos_token标记")
                else:
                    logger.warning(f"警告：聊天模板似乎没有包含预期的bos_token标记 {temp_tokenizer.bos_token}")
                    
            except Exception as e:
                logger.error(f"测试聊天模板时出错: {str(e)}")
                logger.info("将使用回退的模板格式")
                
            # 清理临时tokenizer
            del temp_tokenizer
                
        except Exception as e:
            logger.error(f"聊天模板自检失败: {e}")
            logger.warning("继续训练，但将使用fallback模板。")

    # 正确清理临时accelerator
    # 首先保存分布式状态
    is_initialized = torch.distributed.is_initialized() if hasattr(torch.distributed, 'is_initialized') else False
    
    # 释放临时accelerator内存
    del temp_accelerator
    
    # 如果分布式环境已初始化，则销毁进程组
    if is_initialized:
        try:
            if torch.distributed.is_initialized():
                logger.info("清理临时分布式环境...")
                torch.distributed.destroy_process_group()
        except Exception as e:
            logger.warning(f"清理分布式环境时出错：{str(e)}，继续执行...")
    
    # 加载数据集
    logger.info("加载数据集...")
    dataset = load_dataset(args.data_path)

    # 设置随机种子以保证可复现性
    # random.seed(args.seed)
    # 随机打乱数据集
    random.shuffle(dataset)
    logger.info("数据集已随机打乱。")

    # 构建问题到答案的映射字典，用于直接查询
    qa_dict = {item["question"]: item["answer"] for item in dataset}
    
    # 为数据集创建ID到item的映射，以便reward_fn中直接查询
    dataset_id_map = {}
    for i, item in enumerate(dataset):
        if "id" in item:
            dataset_id_map[item["id"]] = item
        else:
            # 如果没有id，使用索引作为id
            dataset_id_map[str(i)] = item
    
    # 初始化搜索工具
    logger.info("初始化搜索工具...")
    search_tools = SearchTools(
        gnews_api_key=args.gnews_api_key,
        google_api_key=args.google_api_key,
        # proxy_url=args.proxy_url
    )
    
    # 初始化奖励模型
    logger.info("初始化奖励模型...")
    judge = ExternalRewardModel(
        api_key=args.openai_api_key,
        model=args.reward_model,
        score_mode=args.score_mode,
    )
    
    # === 改动部分开始 ===
    # 1. 先加载基座模型 (不含LoRA)
    logger.info(f"加载基础模型: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    base_model.gradient_checkpointing_enable()  # 启用梯度检查点
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer 未定义 pad_token，已设置为 eos_token: {tokenizer.eos_token}")
    
    # 3. 为训练模型配置LoRA
    logger.info("配置LoRA...")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 4. 给训练模型应用LoRA (使用与参考模型同一个基座实例)
    logger.info("给训练模型应用LoRA...")
    train_model = get_peft_model(base_model, lora_cfg)
    train_model.config.use_cache = False     # 训练省显存
    
    # 检查聊天模板特殊标记
    im_start, im_end = get_chat_special_tokens(tokenizer)
    logger.info(f"chat special tokens: start={im_start!r}, end={im_end!r}")
    
    if tokenizer.chat_template is None:
        logger.warning("该 tokenizer 没有 chat_template，所有输入将走 fallback。")
    
    # 初始化生成管理器
    logger.info("初始化生成管理器...")
    generation_manager = GenerationManager(
        tokenizer=tokenizer,
        model=train_model,  # 使用带LoRA的训练模型
        search_tools=search_tools,
        max_turns=args.max_turns,
        system_prompt=SYSTEM_PROMPT_CN,  # 传入系统提示词
    )

    
    # 准备训练数据
    train_dataset = Dataset.from_dict({
        "prompt":    [d["question"]  for d in dataset],
        "reference": [d["answer"]    for d in dataset],
        "sample_id": [str(i)         for i in range(len(dataset))],  # 添加样本ID用于跟踪
    })
    
    # 配置GRPO
    logger.info("配置GRPO训练器...")
    grpo_args = GRPOConfig(
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        output_dir=args.output_dir if args.output_dir else "./results",
        # num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=10,
        logging_steps=1,
        bf16=True,
        # max_grad_norm=1.0,          # ← 开启梯度裁剪
        # warmup_steps=20,
        use_vllm=args.use_vllm, # Pass use_vllm
        # vllm_mode="server", # Default in GRPOConfig
        vllm_server_host=args.vllm_server_host, # Pass vllm_server_host
        vllm_server_port=args.vllm_server_port, # Pass vllm_server_port
        vllm_guided_decoding_regex=r"</search>|</result>",  # 添加</result>作为停止符，防止标签丢失
        report_to="wandb" if args.use_wandb else None,
        remove_unused_columns=False, # Add remove_unused_columns
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # 在Trainer初始化之前定义reward_fn
    def reward_fn(prompts, completions, sample_id=None, **kw):
        """
        符合 trl>=0.18 签名的奖励函数。
        prompts     = 已经包含系统提示、用户查询和助手骨架（思考+搜索）的完整提示部分
        completions = 包含<think><search><result><answer>部分的完成内容
        sample_id   = 样本ID，用于直接从数据集中获取原始问题和参考答案
        返回 1-D tensor/list/numpy array of scores.
        """
        # 引用外部的trainer变量
        nonlocal trainer
        
        # 从sample_id直接获取原始问题和参考答案
        user_questions = []
        references = []
        question_types = []  # 新增：存储问题类型
        
        if sample_id is not None:
            for sid in sample_id:
                # 从数据集获取
                if sid in dataset_id_map:
                    item = dataset_id_map[sid]
                    user_questions.append(item["question"])
                    references.append(item["answer"])
                    # 获取问题类型，默认为"fact"
                    question_types.append(item.get("type", "fact"))
                else:
                    # 如果找不到ID，记录错误并使用空字符串
                    logger.warning(f"无法在数据集中找到ID: {sid}")
                    user_questions.append("")
                    references.append("")
                    question_types.append("fact")
        else:
            # 如果没有提供sample_id，记录警告并使用旧方法
            logger.warning("未提供sample_id，无法从数据集直接获取问题和答案")
            for q_idx, _ in enumerate(prompts):
                if q_idx < len(dataset):
                    user_questions.append(dataset[q_idx]["question"])
                    references.append(dataset[q_idx]["answer"])
                    question_types.append(dataset[q_idx].get("type", "fact"))
                else:
                    user_questions.append("")
                    references.append("")
                    question_types.append("fact")
        
        scores = []
        SEMANTIC_W = 0.5
        DAG_W = 0.25
        R1_W = 0.25
        MAX_SCORE = SEMANTIC_W + DAG_W + R1_W
        EPS = 1e-6

        score_details = [] # 用于wandb日志
        
        # 添加curriculum learning逻辑
        CURRICULUM_STEPS = args.curriculum_steps  # 使用命令行参数
        if trainer and hasattr(trainer, 'state') and trainer.state.global_step < CURRICULUM_STEPS:
            # 前期阶段：让模型专注于格式学习
            SEMANTIC_W, DAG_W, R1_W = 0.2, 0.4, 0.4
            MAX_SCORE = SEMANTIC_W + DAG_W + R1_W
            logger.info(f"Curriculum阶段 (step {trainer.state.global_step}/{CURRICULUM_STEPS}): 使用格式学习奖励权重")
        else:
            # 后期阶段：恢复正常权重
            SEMANTIC_W, DAG_W, R1_W = 0.5, 0.25, 0.25
            MAX_SCORE = SEMANTIC_W + DAG_W + R1_W
            if trainer and hasattr(trainer, 'state') and trainer.state.global_step == CURRICULUM_STEPS:
                logger.info(f"Curriculum结束 (step {trainer.state.global_step}): 恢复标准奖励权重")

        for i, (p, comp, ref, q_type) in enumerate(zip(prompts, completions, references, question_types)):
            # ------------------------------------------------------------
            # 统一提取信息
            # ------------------------------------------------------------
            # full_text   = p + comp                               # 用于格式检查
            user_q      = user_questions[i] if i < len(user_questions) else ""
            answer_match = ANSWER_RE.search(comp)
            answer_text  = answer_match.group(1).strip() if answer_match else comp
            
            logger.info(f"[debug#{i}] user_q={user_q}")
            logger.info(f"[debug#{i}] answer_text={answer_text}")
            logger.info(f"[debug#{i}] ref={ref}")
            logger.info(f"[debug#{i}] type={q_type}")  # 新增：记录问题类型
            # logger.info(f"[debug#{i}] comp={comp} endcomp")
            # logger.info(f"[debug#{i}] p={p} endp")

            # ------------------------------------------------------------
            # 基础格式 / DAG 合法性检查
            # ------------------------------------------------------------
            r1_ok_flag  = check_r1(comp)                    # <think>/<search>/<result>/<answer> 四段完整?
            dag_ok_flag = check_dag(comp)                   # <search> 内 Nodes/Edges 为合法 DAG?
            if not r1_ok_flag:
                logger.info(f"[debug#{i}] comp={comp} endcomp")
            
            logger.info(f"[debug#{i}] r1_ok_flag={r1_ok_flag}")
            logger.info(f"[debug#{i}] dag_ok_flag={dag_ok_flag}")

            # ───────────── 早退：DAG 不合格 ─────────────
            if not dag_ok_flag:
                semantic         = 0.0                           # 不判语义
                dag_ok           = 0.0                           # 失去 DAG_W
                # dag_ok      = -DAG_W                           # 失去 DAG_W
                r1_ok            = R1_W if r1_ok_flag else 0
                # r1_ok            = R1_W if r1_ok_flag else -R1_W                          
                raw_score        = r1_ok
                # normalized_score = max(EPS, min(raw_score / MAX_SCORE, 1.0))
                normalized_score = raw_score
                scores.append(normalized_score)

                score_details.append({
                    "prompt": user_q[:100] + "...",
                    "answer": answer_text[:200] + "...",
                    "semantic_score": semantic,
                    "dag_score": dag_ok,
                    "r1_score": r1_ok,
                    "raw_score": raw_score,
                    "normalized_score": normalized_score,
                })
                continue   # ← 跳到下一条样本
            # ───────────────────────────────────────────

            # ------------------------------------------------------------
            # DAG 合格：继续计算语义得分 + R1 得分
            # ------------------------------------------------------------
            semantic  = judge.semantic_score(user_q, answer_text, ref, q_type)  # 更新：传递问题类型
            dag_ok    = DAG_W                                                  
            # r1_ok     = R1_W if r1_ok_flag else -R1_W                            
            r1_ok     = R1_W if r1_ok_flag else 0
            raw_score = semantic * SEMANTIC_W + dag_ok + r1_ok
            # normalized_score = max(EPS, min(raw_score / MAX_SCORE, 1.0))
            normalized_score = raw_score

            scores.append(normalized_score)

            score_details.append({
                "prompt": user_q[:100] + "...",
                "answer": answer_text[:200] + "...",
                "semantic_score": float(semantic),
                "dag_score": float(dag_ok),
                "r1_score":  float(r1_ok),
                "raw_score": float(raw_score),
                "normalized_score": float(normalized_score),
            })

        # 记录到wandb (如果启用)
        # 判断trainer是否已初始化，若已初始化则使用trainer.accelerator
        is_main_process = False
        if trainer is not None and hasattr(trainer, 'accelerator'):
            is_main_process = trainer.accelerator.is_main_process
        
        if args.use_wandb and is_main_process:
            # 记录本批次的评分详情到wandb
            if len(score_details) > 0:
                # 使用 wandb.Table 记录更详细的信息
                reward_table = wandb.Table(columns=["prompt", "answer", "normalized_score"])
                # 记录第一个样本的详情
                reward_table.add_data(
                    score_details[0]["prompt"],
                    score_details[0]["answer"],
                    score_details[0]["normalized_score"]
                )
                wandb.log({"rewards/sample_details": reward_table})
            
            # 单独记录每个样本的数字指标
            for i, detail in enumerate(score_details):
                wandb.log({
                    f"rewards_{i}/semantic_score": detail["semantic_score"],
                    f"rewards_{i}/dag_score": detail["dag_score"],
                    f"rewards_{i}/r1_score": detail["r1_score"],
                    f"rewards_{i}/raw_score": detail["raw_score"],
                    f"rewards_{i}/normalized_score": detail["normalized_score"]
                })
            
            # 记录批次平均分
            if scores:
                wandb.log({"batch_avg_score": sum(scores) / len(scores)})

        # 返回 1-D scores
        return scores
    
    # 实例化自定义GRPO训练器，传入生成管理器和accelerator，显式传入ref_model
    logger.info("实例化自定义GRPO训练器...")
    trainer = SearchAwareGRPOTrainer(
        model=train_model,
        args=grpo_args,
        processing_class=tokenizer, # ✅ 使用 processing_class
        reward_funcs=reward_fn,   
        train_dataset=train_dataset,
        generation_manager=generation_manager,   # 记得传！
    )

    # 开始训练
    logger.info("开始GRPO训练...")    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # 保存最终模型
    # 使用 trainer 的 accelerator 进行主进程判断
    # if args.output_dir and trainer.accelerator.is_main_process:
    #     os.makedirs(args.output_dir, exist_ok=True)
    #     logger.info(f"保存模型到 {args.output_dir}")
    #     trainer.accelerator.save_model(trainer.model, args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)
        
    # 关闭wandb
    # 使用 trainer 的 accelerator 进行主进程判断
    if args.use_wandb and trainer.accelerator.is_main_process:
        wandb.finish()

    if trainer.accelerator.is_main_process:
        logger.info(f"预训练完成。开始GRPO训练，共{args.epochs}轮...")

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    # 模型与数据参数
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", help="模型名称")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--dataset_path", type=str, default="data/simple_data.jsonl", help="数据集路径")
    parser.add_argument("--save_model", action="store_true", help="是否保存模型")
    parser.add_argument("--save_interval", type=int, default=10, help="保存间隔")
    
    # 奖励模型参数
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API密钥")
    parser.add_argument("--reward_model", type=str, default="gpt-4.1-mini-2025-04-14", help="奖励模型")
    
    # 搜索引擎参数
    parser.add_argument("--gnews_api_key", type=str, default=None, help="GNews API密钥")
    parser.add_argument("--google_api_key", type=str, default=None, help="Google API密钥")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="微批量大小")
    parser.add_argument("--max_prompts", type=int, default=None, help="最大提示数量")
    parser.add_argument("--max_steps", type=int, default=3000, help="最大步数")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="WandB API密钥")
    
    # 添加LoRA相关参数
    parser.add_argument("--lora_r", type=int, default=16, 
                        help="LoRA秩参数")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                        help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout概率")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", 
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
                        help="应用LoRA的目标模块")
    
    # 添加GRPO特定参数
    parser.add_argument("--num_generations", type=int, default=4, 
                        help="GRPO每个查询生成的样本数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--max_turns", type=int, default=2,
                        help="最大生成轮次")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="最大生成长度")
    parser.add_argument("--curriculum_steps", type=int, default=30,
                        help="课程学习阶段的步数，该阶段会更注重格式而非语义。建议值：30-200之间，取决于数据集大小。较大数据集可设置更大值，以确保模型先学习正确的输出格式，再优化语义内容。")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="从指定检查点恢复训练")
                        
    # 添加wandb参数
    parser.add_argument("--use_wandb", action="store_true", 
                        help="是否使用wandb记录训练过程")
    parser.add_argument("--wandb_api_key", type=str, default="", 
                        help="Weights & Biases API密钥")
    parser.add_argument("--wandb_project", type=str, default="rlsf-training",
                        help="Weights & Biases项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases运行名称，默认为rlsf-{model_name}")
    parser.add_argument("--use_vllm", action="store_true",
                        help="是否使用VLLM进行推理加速（注意：需要安装vllm库）")
    parser.add_argument("--vllm_server_host", type=str, default="127.0.0.1",
                        help="vLLM服务器主机地址")
    parser.add_argument("--vllm_server_port", type=int, default=8008,
                        help="vLLM服务器端口")
    
    # 添加奖励模型评分模式参数
    parser.add_argument("--score_mode", type=str, default="continuous", choices=["continuous", "discrete"],
                        help="奖励模型评分模式：'continuous'(连续分数0-1)或'discrete'(离散标签CORRECT/PARTIAL/WRONG)")
    
    args = parser.parse_args()
    
    # 检查API密钥
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    train(args)

if __name__ == "__main__":
    main() 