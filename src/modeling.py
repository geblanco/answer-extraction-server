import os
import json
import torch
import concurrent.futures as cf

from tqdm import tqdm
from functools import partial
from multiprocessing import cpu_count
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)
from transformers.data.processors.squad import (
    SquadExample,
    SquadResult,
    squad_convert_examples_to_features,
    squad_convert_example_to_features,
)

from squad_metrics import compute_predictions_logits

src_path = os.path.abspath(os.path.dirname(__file__))
default_config_path = os.path.join(src_path, 'default_params.json')

NOF_THREADS = min(8, cpu_count())
default_params = json.load(open(default_config_path, 'r'))


class QuestionAnswering(object):

    def find_answers(self, data):
        raise NotImplementedError('You must override `find_answers` method!')

    @classmethod
    def from_pretrained(cls, path_or_name, params=None):
        raise NotImplementedError('You must override `find_answers` method!')

    @staticmethod
    def create_examples(input_data):
        examples = []
        for paragraph in tqdm(
            input_data['paragraphs'],
            desc='Input data to SquadExample'
        ):
            context_text = paragraph['context']
            for qa in paragraph['qas']:
                qas_id = qa['id']
                question_text = qa['question']
                examples.append(SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=None,
                    start_position_character=None,
                    title=None,
                    is_impossible=False,
                    answers=[],
                ))

        return examples

    @staticmethod
    def to_list(tensor):
        return tensor.detach().cpu().tolist()

    @staticmethod
    def features_to_tensors(features):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        return (
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_feature_index,
            all_cls_index,
            all_p_mask,
        )

    @staticmethod
    def convert_examples_to_features(examples, tokenizer, params, nof_threads):
        # ToDo := Fix threading tokenizer (squad converter expects tokenizer to be global)
        # Hint: Extract method from squad.py?
        features = []

        def squad_convert_caller(*args, **kwargs):
            global tokenizer
            return squad_convert_example_to_features(*args, **kwargs)

        with cf.ThreadPoolExecutor(max_workers=nof_threads) as executor:
            annotate_ = partial(
                squad_convert_caller,
                max_seq_length=params['max_seq_length'],
                doc_stride=params['doc_stride'],
                max_query_length=params['max_query_length'],
                is_training=False,
            )
            for feat in executor.map(annotate_, examples):
                features.append(feat)

        return features


class TransformerQuestionAnswering(QuestionAnswering):

    def __init__(self, model, config, tokenizer, params):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.params = params

    def forward_batch_for_results(self, features, batch):
        all_results = []
        self.model.eval()
        batch = tuple(t.to(self.params['device']) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
            }

            if self.config.model_type in ['xlm', 'roberta', 'distilbert', 'camembert']:
                del inputs['token_type_ids']

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if self.config.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(self.model, "config") and hasattr(self.model.config, "lang2id"):
                    lang_id = self.params.get('lang_id', 1)
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * lang_id).to(self.params['device'])}
                    )

            outputs = self.model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            # TODO: i and feature_index are the same number! Simplify by removing enumerate?
            feature = features[feature_index.item()]
            unique_id = int(feature.unique_id)

            output = [QuestionAnswering.to_list(output[i]) for output in outputs]
            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )
            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

        return all_results

    def find_answers_batch(self, data, n_best_size=10, max_answer_length=30):
        """
        GPU Needed?
        Procedure: Convert data to tensor dataset and batch on it
            1. Convert data to SquadExamples, ingestable by transformers
                featurization function
            2. Get a tensor dataset out the examples (along with the features)
            3. Batch process dataset
            4. Join answers
        """
        examples = QuestionAnswering.create_examples(data)
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.params['max_seq_length'],
            doc_stride=self.params['doc_stride'],
            max_query_length=self.params['max_query_length'],
            is_training=False,
            return_dataset="pt",
            threads=NOF_THREADS,
        )
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.params['batch_size']
        )

        all_results = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            results = self.forward_batch_for_results(features, batch)
            all_results.extend(results)

        # If null_score - best_non_null is greater than the
        # threshold predict null.
        threshold = self.params['null_score_diff_threshold']
        predictions, nbest = compute_predictions_logits(
            examples,
            features,
            all_results,
            n_best_size,
            max_answer_length,
            do_lower_case=self.params['do_lower_case'],
            version_2_with_negative=True,
            null_score_diff_threshold=threshold,
            tokenizer=self.tokenizer,
        )
        return predictions, nbest

    def find_answers_simple(self, data, n_best_size=10, max_answer_length=30):
        raise NotImplementedError()
        # # ToDo := Fix threding conversion
        # all_results = []
        # batch_size = self.params['batch_size']
        # examples = QuestionAnswering.create_examples(data)
        # features = QuestionAnswering.convert_examples_to_features(
        #     examples,
        #     self.tokenizer,
        #     self.params,
        #     min(8, cpu_count()),
        # )
        # nof_features = len(features)
        # tensors = QuestionAnswering.features_to_tensors(features)
        # for batch_index in range(0, nof_features, batch_size):
        #     batch_start = batch_index * batch_size
        #     batch_end = min(batch_start + batch_size, nof_features)
        #     batch = tensors[batch_start: batch_end]
        #     result = self.forward_batch_for_results(features, batch)
        #     all_results.append(result)

        # # If null_score - best_non_null is greater than the
        # # threshold predict null.
        # threshold = self.params['null_score_diff_threshold']
        # predictions = compute_predictions_logits(
        #     examples,
        #     features,
        #     all_results,
        #     n_best_size,
        #     max_answer_length,
        #     do_lower_case=self.params['do_lower_case'],
        #     version_2_with_negative=True,
        #     null_score_diff_threshold=threshold,
        #     tokenizer=self.tokenizer,
        # )
        # return predictions

    def find_answers(self, data, n_best_size=1):
        answers, nbest = self.find_answers_batch(data, n_best_size)
        # by now, only batch process on GPU
        # if self.params['device'] != 'cpu' and len(data['paragraphs']) > 50:
        #     # allow default answer len and n_best size
        #     # ToDo := Add to public API
        #     answers = self.find_answers_batch(data)
        # else:
        #     answers = self.find_answers_simple(data)
        first_best = nbest[list(nbest.keys())[0]]
        if n_best_size < len(first_best):
            for key, value in nbest.items():
                length = min(len(nbest[key]), n_best_size)
                nbest[key] = nbest[key][:length]
        return answers, nbest

    @classmethod
    def from_pretrained(cls, path_or_name, params=None):
        config = AutoConfig.from_pretrained(path_or_name)
        tokenizer = AutoTokenizer.from_pretrained(path_or_name)
        model = AutoModelForQuestionAnswering.from_config(config)
        user_params = default_params.copy()
        if params is not None:
            user_params.update(**params)
        return cls(model, config, tokenizer, user_params)
