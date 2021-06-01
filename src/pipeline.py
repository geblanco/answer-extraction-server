from transformers import pipeline


class QuestionAnswering(object):
    def __init__(self, model_path):
        self._model_path = model_path
        self.pipeline = pipeline("question-answering", model=model_path)

    def find_answers(self, data, topk=1):
        answers = []
        # see data/formatted/squad/dev-v2.0_formatted_10_decoded.json
        for elem in data:
            qas, ctx, cid = elem.values()
            qas_out = {"qas": [], "cid": cid}
            questions = [q["question"] for q in qas]
            results = self.pipeline(question=questions, context=ctx, topk=topk)
            res_by_qs = [results[i:i + 4] for i in range(0, len(results), topk)]
            for q_triplet, res in zip(qas, res_by_qs):
                qas_out["qas"].append({
                    "qid": q_triplet["qid"],
                    "id": q_triplet["id"],
                    "results": [{
                        "text": ans_res["answer"],
                        "score": round(ans_res["score"], 4),
                        "start": ans_res["start"],
                        "end": ans_res["end"]
                    } for ans_res in res]
                })
            answers.append(qas_out)

        return answers

