import os
import json
import requests
from collections import defaultdict
from typing import List
import pickle
from tqdm import tqdm
import re  # <-- THIS IS THE FIX

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
        """
        if not dataset_path:
            raise ValueError("必须提供数据集路径 (dataset_path)。")

        dataset_path = os.path.abspath(os.path.expanduser(dataset_path))

        self.execution_model = execution_model
        self.ollama_api_url = f"{ollama_base_url}/api/generate"
        self.temperature = temperature
        self.device = device
        self.reward_cache_path = os.path.join(dataset_path, 'rewards.pkl')

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
            # I've also re-added the timeout for robustness
            response = requests.post(self.ollama_api_url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json().get('response', '')
        except requests.RequestException as e:
            print(f"调用Ollama模型 {model_name} 时出错: {e}")
            return ""

    def predict_tail_entities_with_multi_agents(self, head_id: int, relation_id: int, timestamp: int,
                                                rl_candidate_entities: List[str], top_k: int = 10) -> List[tuple]:
        """
        使用LLM根据RL提供的候选实体列表进行最终预测。
        """
        head_text = self.id_to_entity.get(head_id, f"实体_{head_id}")
        relation_text = self.id_to_relation.get(relation_id, f"关系_{relation_id}")
        date_text = self.id_to_ts.get(timestamp, f"时间_{timestamp}")

        available_entities_for_prompt = "\n".join([f"- {name}" for name in rl_candidate_entities])
        history_facts = self._get_recent_history(head_id, timestamp)
        history_text = self._convert_facts_to_text(history_facts)
        objective = f"在{date_text}，当发生事件“{head_text} {relation_text} ?”时，最有可能的尾实体是什么？"

        final_prediction_prompt = f"""You are an expert in temporal knowledge graph reasoning. Your task is to predict the most likely tail entity based on the provided context and a list of high-quality candidates.

### Objective:
{objective}

### Historical Context (Recent events involving "{head_text}"):
{history_text}

### High-Quality Candidate Entities (Your final answer MUST be strictly selected from this list):
{available_entities_for_prompt}

### Instructions:
1. Analyze the "Historical Context" and the "Objective" to understand logical and temporal patterns.
2. Your answer MUST be one of the entities from the "High-Quality Candidate Entities" list.
3. List the top {top_k} most likely entities, ordered from most likely to least likely.
4. Provide ONLY the ordered list of entity names. Do not include any explanations or reasoning.

### Final Prediction:
"""
        final_result = self._call_ollama_model(self.execution_model, final_prediction_prompt)
        return self._parse_final_predictions(final_result, top_k, rl_candidate_entities)

    def _parse_final_predictions(self, response_text: str, top_k: int, available_entities: List[str]) -> List[tuple]:
        """
        解析LLM的输出，提取排序后的实体列表。
        """
        predictions = []
        lines = [line.strip() for line in response_text.strip().split('\n')]

        for line in lines:
            if not line: continue
            entity_text = line.split('.', 1)[-1].strip().replace('*', '').replace('-', '').strip()
            matched_entity = next((cand for cand in available_entities if cand.lower() == entity_text.lower()), None)
            if matched_entity:
                if matched_entity not in [p[1] for p in predictions]:
                    predictions.append(("llm_prediction", matched_entity))
            if len(predictions) >= top_k:
                break

        if len(predictions) < top_k:
            for cand in available_entities:
                if cand not in [p[1] for p in predictions]:
                    predictions.append(("rl_candidate_fallback", cand))
                if len(predictions) >= top_k:
                    break
        return predictions

    def generate_and_cache_rewards(self, dataloader):
        """
        【优化版】使用LLM从训练数据生成奖励并缓存到文件。
        该版本采用批处理方式，将一个批次的数据合并为单个Prompt进行请求。
        """
        rewards = {}
        processed_count = 0

        for batch in tqdm(dataloader, desc="Generating LLM Rewards (Batch Mode)"):
            src_batch, rel_batch, dst_batch, time_batch = batch
            batch_prompt_parts = []

            for i in range(len(src_batch)):
                head_id = src_batch[i].item()
                relation_id = rel_batch[i].item()
                timestamp = time_batch[i].item()

                head_text = self.id_to_entity.get(head_id, f"实体_{head_id}")
                relation_text = self.id_to_relation.get(relation_id, f"关系_{relation_id}")
                date_text = self.id_to_ts.get(timestamp, f"时间_{timestamp}")

                history_facts = self._get_recent_history(head_id, timestamp)
                history_text = self._convert_facts_to_text(history_facts)
                objective = f"在{date_text}，当发生事件“{head_text} {relation_text} ?”时，最有可能的尾实体是什么？"

                prompt_part = f"""--- Sample {i} ---
### Objective:
{objective}

### Historical Context:
{history_text}
"""
                batch_prompt_parts.append(prompt_part)

            final_batch_prompt = "You will be given multiple samples, each with an objective and historical context. For each sample, predict the single most likely tail entity.\nYour response must follow this format exactly, with one line for each sample:\nAnswer for Sample 0: [Entity Name]\nAnswer for Sample 1: [Entity Name]\n...\n\n" + "\n".join(
                batch_prompt_parts)

            llm_batch_response = self._call_ollama_model(self.execution_model, final_batch_prompt)

            if not llm_batch_response:
                print(f"  - Skipped batch due to LLM call failure.")
                continue

            predictions = re.findall(r"Answer for Sample (\d+):\s*(.*)", llm_batch_response)
            parsed_predictions = {int(index): answer.strip() for index, answer in predictions}

            for i in range(len(src_batch)):
                llm_prediction = parsed_predictions.get(i)
                if llm_prediction is None:
                    continue

                true_tail_id = dst_batch[i].item()
                true_tail_text = self.id_to_entity.get(true_tail_id, f"实体_{true_tail_id}")
                reward = 1.0 if true_tail_text.lower() in llm_prediction.lower() else -1.0
                cache_key = (src_batch[i].item(), rel_batch[i].item(), time_batch[i].item())
                rewards[cache_key] = reward
                processed_count += 1

        print(f"\n成功处理了 {processed_count} 个样本。")
        with open(self.reward_cache_path, 'wb') as f:
            pickle.dump(rewards, f)
        print(f"奖励已生成并缓存到 {self.reward_cache_path}")

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
        past_facts = [fact for fact in related_facts if fact[2] < timestamp]
        past_facts.sort(key=lambda x: x[2], reverse=True)
        return past_facts[:limit]

    def _convert_facts_to_text(self, facts: List[tuple]) -> str:
        """将事实元组转换为可读的文本格式。"""
        if not facts:
            return "No recent historical events found."

        history_text = ""
        for r, t, ts in facts:
            r_text = self.id_to_relation.get(r, f"关系_{r}")
            t_text = self.id_to_entity.get(t, f"实体_{t}")
            date_text = self.id_to_ts.get(ts, f"时间_{ts}")
            history_text += f"- On {date_text}, event involved {t_text} via relation {r_text}.\n"
        return history_text