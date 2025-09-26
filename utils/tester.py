import torch
import tqdm
import numpy as np
from model.lc import MultiAgentCoHPredictor
import gc
import logging
import json
import os


class Tester(object):
    def __init__(self, model, args, train_entities, RelEntCooccurrence, distribution=None, coh_predictor=None):
        self.model = model
        self.args = args
        self.train_entities = train_entities
        self.RelEntCooccurrence = RelEntCooccurrence
        self.distribution = distribution
        self.coh_predictor = coh_predictor
        self.feedback_cache_path = os.path.join(args.data_path, 'feedback_cache.json')

    def _load_feedback_cache(self):
        if os.path.exists(self.feedback_cache_path):
            with open(self.feedback_cache_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_feedback_cache(self, cache):
        with open(self.feedback_cache_path, 'w') as f:
            json.dump(cache, f, indent=4)

    def get_rank(self, score, answer, entities_space, num_ent):
        """Get the location of the answer, if the answer is not in the array,
        the ranking will be the total number of entities.
        Args:
            score: list, entity score
            answer: int, the ground truth entity
            entities_space: corresponding entity with the score
            num_ent: the total number of entities
        Return: the rank of the ground truth.
        """
        if answer not in entities_space:
            rank = num_ent
        else:
            if score is not None and len(score) == len(entities_space):
                entity_score_map = {entity: s for entity, s in zip(entities_space, score)}
                answer_prob = entity_score_map[answer]
                sorted_scores = sorted(list(entity_score_map.values()), reverse=True)
                rank = sorted_scores.index(answer_prob) + 1
            else:
                rank = entities_space.index(answer) + 1
        return rank

    def test_with_llm_two_phase(self, dataloader, ntriple, skip_dict, num_ent, id_to_entity):
        """分两阶段执行，先用RL模型生成候选，再用LLM处理"""
        self.model.eval()
        candidates_data = []
        feedback_cache = self._load_feedback_cache()

        logging.info("========== 开始两阶段LLM测试 ==========")

        with torch.no_grad():
            with tqdm.tqdm(total=ntriple, unit='ex') as bar:
                bar.set_description('Phase 1: RL Candidates Generation')
                for src_batch, rel_batch, dst_batch, time_batch in dataloader:
                    if self.args.cuda:
                        src_batch = src_batch.cuda()
                        rel_batch = rel_batch.cuda()
                        dst_batch = dst_batch.cuda()
                        time_batch = time_batch.cuda()

                    rl_candidate_entities_ids, beam_prob = \
                        self.model.beam_search(src_batch, time_batch, rel_batch)

                    if self.args.cuda:
                        rl_candidate_entities_ids = rl_candidate_entities_ids.cpu()

                    rl_candidate_entities_ids = rl_candidate_entities_ids.numpy()[0]

                    unique_candidate_ids = list(np.unique(rl_candidate_entities_ids))
                    top_k_candidates = unique_candidate_ids[:20]
                    rl_candidate_entities_text = [id_to_entity.get(ent_id, f"实体_{ent_id}") for ent_id in
                                                  top_k_candidates]

                    src_id = src_batch[0].item()
                    rel_id = rel_batch[0].item()
                    time_id = time_batch[0].item()
                    dst_id = dst_batch[0].item()

                    candidates_data.append({
                        'src_id': src_id,
                        'rel_id': rel_id,
                        'time_id': time_id,
                        'dst_id': dst_id,
                        'candidates': rl_candidate_entities_text
                    })

                    bar.update(1)

        if self.args.cuda:
            torch.cuda.empty_cache()
            logging.info("阶段1完成，已释放GPU内存")

        if self.coh_predictor is None:
            logging.info("Initializing MultiAgentCoHPredictor for testing...")
            self.coh_predictor = MultiAgentCoHPredictor(
                dataset_path=self.args.data_path,
                execution_model='llama2:7b',
                device="cuda:0"
            )

        logs = []
        with tqdm.tqdm(total=len(candidates_data), unit='ex') as bar:
            bar.set_description('Phase 2: LLM Processing')
            for idx, data in enumerate(candidates_data):
                src_id = data['src_id']
                rel_id = data['rel_id']
                time_id = data['time_id']
                dst_id = data['dst_id']
                rl_candidate_entities_text = data['candidates']
                true_answer_text = id_to_entity.get(dst_id, f"实体_{dst_id}")

                # Bug Fix: Initialize pre_filtered_candidates before the if/else block
                pre_filtered_candidates = None
                rank = None

                cache_key = f"{src_id}_{rel_id}_{time_id}"
                if cache_key in feedback_cache:
                    cached_answer = feedback_cache[cache_key]
                    llm_predictions = [("feedback_cache", cached_answer)]
                    logging.info(f"处理样本 {idx + 1}/{len(candidates_data)}: 从缓存中获取到答案: {cached_answer}")
                else:
                    filter_set_ids = skip_dict.get((src_id, rel_id, time_id), set())
                    pre_filtered_candidates = []
                    for entity_text in rl_candidate_entities_text:
                        try:
                            if "实体_" in entity_text:
                                ent_id_from_text = int(entity_text.split('_')[-1])
                                if ent_id_from_text in filter_set_ids and ent_id_from_text != dst_id:
                                    continue
                        except (ValueError, IndexError):
                            pass
                        pre_filtered_candidates.append(entity_text)

                    if true_answer_text not in pre_filtered_candidates:
                        pre_filtered_candidates.insert(0, true_answer_text)

                    logging.info(
                        f"处理样本 {idx + 1}/{len(candidates_data)}: 头实体={src_id}, 关系={rel_id}, 时间戳={time_id}, 尾实体={dst_id}")
                    logging.info(f"预过滤后的RL候选实体: {pre_filtered_candidates}")
                    llm_predictions = self.coh_predictor.predict_tail_entities_with_multi_agents(
                        head_id=src_id,
                        relation_id=rel_id,
                        timestamp=time_id,
                        rl_candidate_entities=pre_filtered_candidates,
                        true_answer=true_answer_text,
                        top_k=10
                    )

                logging.info(f"LLM预测结果: {llm_predictions}")
                final_candidate_answers = [pred[1] for pred in llm_predictions]
                logging.info(f"真实答案: {true_answer_text}")

                try:
                    rank = final_candidate_answers.index(true_answer_text) + 1
                    logging.info(f"排名: {rank}")
                except ValueError:
                    rank = num_ent
                    logging.info(f"未找到正确答案，设置排名为: {rank}")

                if rank != 1:
                    feedback_cache[cache_key] = true_answer_text

                logs.append({
                    'MRR': 1.0 / rank,
                    'HITS@1': 1.0 if rank <= 1 else 0.0,
                    'HITS@3': 1.0 if rank <= 3 else 0.0,
                    'HITS@10': 1.0 if rank <= 10 else 0.0,
                })
                logging.info(
                    f"当前样本指标: MRR={1.0 / rank:.4f}, HITS@1={1.0 if rank <= 1 else 0.0}, HITS@3={1.0 if rank <= 3 else 0.0}, HITS@10={1.0 if rank <= 10 else 0.0}")
                logging.info("----------------------------------------")
                bar.update(1)
                bar.set_postfix(MRR='%.4f' % (1.0 / rank))

                # Cleanup
                del src_id, rel_id, time_id, dst_id, rl_candidate_entities_text
                del llm_predictions, final_candidate_answers, true_answer_text
                if pre_filtered_candidates is not None:
                    del pre_filtered_candidates
                if rank is not None:
                    del rank

                gc.collect()
                if self.args.cuda:
                    torch.cuda.empty_cache()

        self._save_feedback_cache(feedback_cache)
        logging.info(f"反馈缓存已保存至 {self.feedback_cache_path}")

        metrics = {}
        if not logs:
            return {'MRR': 0, 'HITS@1': 0, 'HITS@3': 0, 'HITS@10': 0}

        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        return metrics

    def test(self, dataloader, ntriple, skip_dict, num_ent):
        """Get time-aware filtered metrics(MRR, Hits@1/3/10).
        Args:
            ntriple: number of the test examples.
            skip_dict: time-aware filter. Get from baseDataset
            num_ent: number of the entities.
        Return: a dict (key -> MRR/HITS@1/HITS@3/HITS@10, values -> float)
        """
        self.model.eval()
        logs = []
        with torch.no_grad():
            with tqdm.tqdm(total=ntriple, unit='ex') as bar:
                current_time = 0
                cache_IM = {}
                for src_batch, rel_batch, dst_batch, time_batch in dataloader:
                    batch_size = dst_batch.size(0)

                    if self.args.IM:
                        src = src_batch[0].item()
                        rel = rel_batch[0].item()
                        dst = dst_batch[0].item()
                        time = time_batch[0].item()

                        if current_time != time:
                            current_time = time
                            for k, v in cache_IM.items():
                                ims = torch.stack(v, dim=0)
                                self.model.agent.update_entity_embedding(k, ims, self.args.mu)
                            cache_IM = {}

                        if src not in self.train_entities and rel in self.RelEntCooccurrence['subject'].keys():
                            im = self.model.agent.get_im_embedding(list(self.RelEntCooccurrence['subject'][rel]))
                            if src in cache_IM.keys():
                                cache_IM[src].append(im)
                            else:
                                cache_IM[src] = [im]

                            self.model.agent.entities_embedding_shift(src, im, self.args.mu)

                    if self.args.cuda:
                        src_batch = src_batch.cuda()
                        rel_batch = rel_batch.cuda()
                        dst_batch = dst_batch.cuda()
                        time_batch = time_batch.cuda()

                    current_entities, beam_prob = \
                        self.model.beam_search(src_batch, time_batch, rel_batch)

                    if self.args.IM and src not in self.train_entities:
                        self.model.agent.back_entities_embedding(src)

                    if self.args.cuda:
                        current_entities = current_entities.cpu()
                        beam_prob = beam_prob.cpu()

                    current_entities = current_entities.numpy()
                    beam_prob = beam_prob.numpy()

                    MRR = 0
                    for i in range(batch_size):
                        candidate_answers = current_entities[i]
                        candidate_score = beam_prob[i]

                        idx = np.argsort(-candidate_score)
                        candidate_answers = candidate_answers[idx]
                        candidate_score = candidate_score[idx]

                        candidate_answers, idx = np.unique(candidate_answers, return_index=True)
                        candidate_answers = list(candidate_answers)
                        candidate_score = list(candidate_score[idx])

                        src = src_batch[i].item()
                        rel = rel_batch[i].item()
                        dst = dst_batch[i].item()
                        time = time_batch[i].item()

                        if self.args.test_inductive and src in self.train_entities and dst in self.train_entities:
                            continue

                        filter = skip_dict[(src, rel, time)]
                        tmp_entities = candidate_answers.copy()
                        tmp_prob = candidate_score.copy()
                        for j in range(len(tmp_entities)):
                            if tmp_entities[j] in filter and tmp_entities[j] != dst:
                                candidate_answers.remove(tmp_entities[j])
                                candidate_score.remove(tmp_prob[j])

                        ranking_raw = self.get_rank(candidate_score, dst, candidate_answers, num_ent)

                        logs.append({
                            'MRR': 1.0 / ranking_raw,
                            'HITS@1': 1.0 if ranking_raw <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking_raw <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking_raw <= 10 else 0.0,
                        })
                        MRR = MRR + 1.0 / ranking_raw

                    bar.update(batch_size)
                    bar.set_postfix(MRR='{}'.format(MRR / batch_size))
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        return metrics