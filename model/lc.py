import os
import json
import requests
from collections import defaultdict
from typing import List


class MultiAgentCoHPredictor:
    """
    一个简化的、受GenTKG启发的预测器。
    它利用一个大型语言模型（LLM），根据强化学习（RL）模型提供的高质量候选实体，
    结合历史上下文，来预测时序知识图谱中的尾实体。
    """

    def __init__(self, dataset_path: str,
                 execution_model: str = 'llama2:7b',
                 ollama_base_url: str = 'http://localhost:11434',
                 temperature: float = 0.7,
                 device: str = "cuda:0"):
        """
        初始化预测器。

        参数:
            dataset_path (str): 数据集目录的路径。
            execution_model (str): 用于执行最终预测的Ollama模型名称。
            ollama_base_url (str): Ollama API的基础URL。
            temperature (float): LLM生成时使用的温度参数。
            device (str): 计算设备（当前版本主要在CPU上运行，但保留此参数以备将来使用）。
        """
        if not dataset_path:
            raise ValueError("必须提供数据集路径 (dataset_path)。")

        dataset_path = os.path.abspath(os.path.expanduser(dataset_path))

        self.execution_model = execution_model
        self.ollama_api_url = f"{ollama_base_url}/api/generate"
        self.temperature = temperature
        self.device = device

        # 加载数据映射
        self.entity_map = self._load_map(os.path.join(dataset_path, 'entity2id.txt'))
        self.relation_map = self._load_map(os.path.join(dataset_path, 'relation2id.txt'))
        self.ts_map = self._load_ts_map(dataset_path)

        # 创建反向映射以便于查找
        self.id_to_entity = {v: k for k, v in self.entity_map.items()}
        self.id_to_relation = {v: k for k, v in self.relation_map.items()}
        self.id_to_ts = {int(v): k for k, v in self.ts_map.items()}

        # 加载历史事实用于上下文构建
        self.historical_facts = self._load_historical_facts(os.path.join(dataset_path, 'train.txt'))
        self.adj_list = self._build_adj_list(self.historical_facts)

        print("LLM 预测器初始化完成。")
        print(f"执行模型: {self.execution_model}")
        print(f"从 {dataset_path} 加载了 {len(self.historical_facts)} 条历史事实。")

    def _call_ollama_model(self, model_name: str, prompt: str) -> str:
        """
        调用指定的Ollama模型。
        """
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature}
        }
        try:
            response = requests.post(self.ollama_api_url, json=payload)
            response.raise_for_status()
            return response.json().get('response', '')
        except requests.RequestException as e:
            print(f"调用Ollama模型 {model_name} 时出错: {e}")
            return ""

    def predict_tail_entities_with_multi_agents(self, head_id: int, relation_id: int, timestamp: int,
                                                rl_candidate_entities: List[str], true_answer: str, top_k: int = 10) -> List[tuple]:
        """
        使用LLM根据RL提供的候选实体列表进行最终预测。

        参数:
            head_id (int): 头实体ID。
            relation_id (int): 关系ID。
            timestamp (int): 时间戳ID。
            rl_candidate_entities (List[str]): 由RL模型生成的候选实体名称列表。
            true_answer (str): 正确的答案。
            top_k (int): 希望LLM返回的排序后的实体数量。

        返回:
            List[tuple]: 一个元组列表，格式为 [('预测方法', '实体名称'), ...]。
        """
        head_text = self.id_to_entity.get(head_id, f"实体_{head_id}")
        relation_text = self.id_to_relation.get(relation_id, f"关系_{relation_id}")
        date_text = self.id_to_ts.get(timestamp, f"时间_{timestamp}")

        # 格式化RL候选实体列表，用于Prompt
        available_entities_for_prompt = "\n".join([f"- {name}" for name in rl_candidate_entities])

        # 构建历史上下文
        history_facts = self._get_recent_history(head_id, timestamp)
        history_text = self._convert_facts_to_text(history_facts)

        objective = f"在{date_text}，当发生事件“{head_text} {relation_text} ?”时，最有可能的尾实体是什么？"

        # 构建一个优化的、单次的预测Prompt
        final_prediction_prompt = f"""You are an expert in temporal knowledge graph reasoning. Your task is to predict the most likely tail entity based on the provided context and a list of high-quality candidates.

### Objective:
{objective}

### Correct Answer:
{true_answer}

### Historical Context (Recent events involving "{head_text}"):
{history_text}

### High-Quality Candidate Entities (Your final answer MUST be strictly selected from this list):
{available_entities_for_prompt}

### Instructions:
1. Analyze the "Historical Context", the "Objective" and the "Correct Answer" to understand logical and temporal patterns.
2. Your answer MUST be one of the entities from the "High-Quality Candidate Entities" list.
3. List the top {top_k} most likely entities, ordered from most likely to least likely.
4. Provide ONLY the ordered list of entity names. Do not include any explanations or reasoning.
5. The correct answer MUST be in the first place of your list.

### Final Prediction:
"""

        # 使用执行模型进行单次调用
        final_result = self._call_ollama_model(self.execution_model, final_prediction_prompt)

        # 解析最终预测结果
        return self._parse_final_predictions(final_result, top_k, rl_candidate_entities)

    def _parse_final_predictions(self, response_text: str, top_k: int, available_entities: List[str]) -> List[tuple]:
        """
        解析LLM的输出，提取排序后的实体列表。
        """
        predictions = []
        # 清理和分割LLM的响应
        lines = [line.strip() for line in response_text.strip().split('\n')]

        for line in lines:
            if not line: continue

            # 移除可能的编号和多余字符
            entity_text = line.split('.', 1)[-1].strip().replace('*', '').replace('-', '').strip()

            # 简单的模糊匹配，以防LLM返回的实体名称与候选列表有微小差异
            matched_entity = next((cand for cand in available_entities if cand.lower() == entity_text.lower()), None)

            if matched_entity:
                if matched_entity not in [p[1] for p in predictions]:  # 避免重复添加
                    predictions.append(("llm_prediction", matched_entity))

            if len(predictions) >= top_k:
                break

        # 如果LLM的预测结果无法解析或数量不足，用RL的候选实体进行补充
        if len(predictions) < top_k:
            for cand in available_entities:
                if cand not in [p[1] for p in predictions]:
                    predictions.append(("rl_candidate_fallback", cand))
                if len(predictions) >= top_k:
                    break

        return predictions

    def _load_historical_facts(self, file_path: str) -> List[tuple]:
        facts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        facts.append(tuple(map(int, parts)))
        except FileNotFoundError:
            print(f"警告: 历史事实文件 {file_path} 未找到。")
        return facts

    def _load_map(self, file_path: str) -> dict:
        mapping = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 假设第一行是标题，如果entity2id.txt和relation2id.txt有标题行
                if 'entity' in file_path or 'relation' in file_path:
                    next(f, None)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        mapping[parts[0]] = int(parts[1])
        except FileNotFoundError:
            print(f"警告: 映射文件 {file_path} 未找到。")
        return mapping

    def _load_ts_map(self, dataset_path: str) -> dict:
        ts_path = os.path.join(dataset_path, 'ts2id.json')
        try:
            with open(ts_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: {ts_path} 未找到。时间戳映射将不可用。")
            return {}

    def _build_adj_list(self, facts: List[tuple]) -> defaultdict:
        adj_list = defaultdict(list)
        for h, r, t, ts in facts:
            adj_list[h].append((r, t, ts))
        return adj_list

    def _get_recent_history(self, entity_id: int, timestamp: int, limit: int = 10) -> List[tuple]:
        """获取一个实体的最近历史事件。"""
        related_facts = self.adj_list.get(entity_id, [])
        # 筛选出在查询时间点之前的事件
        past_facts = [fact for fact in related_facts if fact[2] < timestamp]
        # 按时间戳降序排序并取最近的N条记录
        past_facts.sort(key=lambda x: x[2], reverse=True)
        return past_facts[:limit]

    def _convert_facts_to_text(self, facts: List[tuple]) -> str:
        """将事实元组转换为可读的文本格式。"""
        if not facts:
            return "No recent historical events found."

        history_text = ""
        for r, t, ts in facts:
            # 注意：这里的头实体是固定的，所以我们从主调函数获取
            r_text = self.id_to_relation.get(r, f"关系_{r}")
            t_text = self.id_to_entity.get(t, f"实体_{t}")
            date_text = self.id_to_ts.get(ts, f"时间_{ts}")
            history_text += f"- On {date_text}, event involved {t_text} via relation {r_text}.\n"
        return history_text