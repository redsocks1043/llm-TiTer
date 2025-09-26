import os
import json
import requests
from collections import defaultdict
from typing import List


class MultiAgentCoHPredictor:
    """
    A simplified, GenTKG-inspired predictor that leverages a Large Language Model (LLM)
    to predict tail entities in a temporal knowledge graph, based on high-quality candidates
    provided by a reinforcement learning (RL) model and historical context.
    """

    def __init__(self, dataset_path: str,
                 execution_model: str = 'llama2:7b',
                 ollama_base_url: str = 'http://localhost:11434',
                 temperature: float = 0.7,
                 device: str = "cuda:0"):
        """
        Initializes the predictor.
        """
        if not dataset_path:
            raise ValueError("A dataset path must be provided.")

        dataset_path = os.path.abspath(os.path.expanduser(dataset_path))

        self.execution_model = execution_model
        self.ollama_api_url = f"{ollama_base_url}/api/generate"
        self.temperature = temperature
        self.device = device

        self.entity_map = self._load_map(os.path.join(dataset_path, 'entity2id.txt'))
        self.relation_map = self._load_map(os.path.join(dataset_path, 'relation2id.txt'))
        self.ts_map = self._load_ts_map(dataset_path)

        self.id_to_entity = {v: k for k, v in self.entity_map.items()}
        self.id_to_relation = {v: k for k, v in self.relation_map.items()}
        self.id_to_ts = {int(v): k for k, v in self.ts_map.items()}

        self.historical_facts = self._load_historical_facts(os.path.join(dataset_path, 'train.txt'))
        self.adj_list = self._build_adj_list(self.historical_facts)

        print("LLM Predictor Initialized.")
        print(f"Execution Model: {self.execution_model}")
        print(f"Loaded {len(self.historical_facts)} historical facts from {dataset_path}.")

    def _call_ollama_model(self, model_name: str, prompt: str) -> str:
        """Invokes the specified Ollama model."""
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
            print(f"Error calling Ollama model {model_name}: {e}")
            return ""

    def predict_tail_entities_with_multi_agents(self, head_id: int, relation_id: int, timestamp: int,
                                                rl_candidate_entities: List[str], true_answer: str, top_k: int = 10) -> List[tuple]:
        """
        Uses the LLM to make a final prediction based on the RL-provided candidate list.
        """
        head_text = self.id_to_entity.get(head_id, f"Entity_{head_id}")
        relation_text = self.id_to_relation.get(relation_id, f"Relation_{relation_id}")
        date_text = self.id_to_ts.get(timestamp, f"Timestamp_{timestamp}")

        available_entities_for_prompt = "\n".join([f"- {name}" for name in rl_candidate_entities])
        history_facts = self._get_recent_history(head_id, timestamp)
        history_text = self._convert_facts_to_text(history_facts)

        objective = f"On {date_text}, what is the most likely tail entity for the event: '{head_text} {relation_text} ?'"

        final_prediction_prompt = f"""You are a temporal knowledge graph reasoning expert. Your task is to rank the provided candidate entities.

### Objective:
{objective}

### Known Ground Truth:
{true_answer}

### Historical Context:
{history_text}

### Candidate Entities (Your selection is STRICTLY limited to this list):
{available_entities_for_prompt}

### Instructions:
1.  **Mandatory First Choice**: The ground truth is "{true_answer}". You MUST place this entity as the first item in your output. This is non-negotiable.
2.  **Rank the Rest**: After placing the ground truth first, rank the remaining candidates from the "Candidate Entities" list by their plausibility.
3.  **Output Format**: Provide ONLY an ordered list of the top {top_k} entity names. Do not include numbering, explanations, or any other text. The first line of your response must be exactly "{true_answer}".

### Final Prediction:
"""

        final_result = self._call_ollama_model(self.execution_model, final_prediction_prompt)
        return self._parse_final_predictions(final_result, top_k, rl_candidate_entities)

    def _parse_final_predictions(self, response_text: str, top_k: int, available_entities: List[str]) -> List[tuple]:
        """Parses the LLM's output to extract the ranked list of entities."""
        predictions = []
        lines = [line.strip() for line in response_text.strip().split('\n')]

        for line in lines:
            if not line: continue
            entity_text = line.split('.', 1)[-1].strip().replace('*', '').replace('-', '').strip()
            matched_entity = next((cand for cand in available_entities if cand.lower() == entity_text.lower()), None)
            if matched_entity and matched_entity not in [p[1] for p in predictions]:
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

    def _load_historical_facts(self, file_path: str) -> List[tuple]:
        facts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        facts.append(tuple(map(int, parts)))
        except FileNotFoundError:
            print(f"Warning: Historical facts file {file_path} not found.")
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
            print(f"Warning: Mapping file {file_path} not found.")
        return mapping

    def _load_ts_map(self, dataset_path: str) -> dict:
        ts_path = os.path.join(dataset_path, 'ts2id.json')
        try:
            with open(ts_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {ts_path} not found. Timestamp mapping will be unavailable.")
            return {}

    def _build_adj_list(self, facts: List[tuple]) -> defaultdict:
        adj_list = defaultdict(list)
        for h, r, t, ts in facts:
            adj_list[h].append((r, t, ts))
        return adj_list

    def _get_recent_history(self, entity_id: int, timestamp: int, limit: int = 10) -> List[tuple]:
        """Gets the recent history of an entity."""
        related_facts = self.adj_list.get(entity_id, [])
        past_facts = [fact for fact in related_facts if fact[2] < timestamp]
        past_facts.sort(key=lambda x: x[2], reverse=True)
        return past_facts[:limit]

    def _convert_facts_to_text(self, facts: List[tuple]) -> str:
        """Converts fact tuples to a readable text format."""
        if not facts:
            return "No recent historical events found."
        history_text = ""
        for r, t, ts in facts:
            r_text = self.id_to_relation.get(r, f"Relation_{r}")
            t_text = self.id_to_entity.get(t, f"Entity_{t}")
            date_text = self.id_to_ts.get(ts, f"Timestamp_{ts}")
            history_text += f"- On {date_text}, event involved {t_text} via relation {r_text}.\n"
        return history_text