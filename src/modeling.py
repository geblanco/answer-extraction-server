import json
import torch


from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    squad_convert_examples_to_features,
)
from transformers.data.processors.squad import (
    SquadExample,
    SquadResult,
)

from transformers.data.metrics.squad_metrics import compute_predictions_logits

NOF_THREADS = 8
default_params = json.load(open('./default_params.json', 'r'))


class QuestionAnswering(object):

    def find_answers(self, data):
        raise NotImplementedError('You must override `find_answers` method!')

    def to_list(tensor):
        return tensor.detach().cpu().tolist()

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


class TransformerQuestionAnswering(QuestionAnswering):

    def __init__(self, model, tokenizer, params):
        self.model = model
        self.tokenizer = tokenizer
        self.params = params

    def forward_batch_for_logits(self, features, batch):
        self.model.eval()
        batch = tuple(t.to(self.params['device']) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
            }

            if self.params['model_type'] in ['xlm', 'roberta', 'distilbert', 'camembert']:
                del inputs['token_type_ids']

            feature_indices = batch[3]
            outputs = self.model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            # TODO: i and feature_index are the same number! Simplify by removing enumerate?
            feature = features[feature_index.item()]
            unique_id = int(feature.unique_id)

            output = [self.to_list(output[i]) for output in outputs]
            start_logits, end_logits = output

        return unique_id, start_logits, end_logits

    def find_answers_batch(self, data, n_best_size=5, max_answer_length=30):
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
            unique_id, start_logits, end_logits = \
                self.forward_batch_for_logits(features, batch)
            all_results.append(SquadResult(unique_id, start_logits, end_logits))

        # If null_score - best_non_null is greater than the
        # threshold predict null.
        threshold = self.params['null_score_diff_threshold']
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            n_best_size,
            max_answer_length,
            do_lower_case=self.params['do_lower_case'],
            verbose_logging=True,
            version_2_with_negative=True,
            null_score_diff_threshold=threshold,
            tokenizer=self.tokenizer,
        )
        return predictions

    def find_answers_simple(self, data):
        answers = []
        self.model.eval()
        for par in data['paragraphs']:
            context = par['context']
            for qa in par['qas']:
                # ToDo := tokenizer options, crop, etc
                qa_enc = self.tokenizer.encode_plus(context, qa['question'])
                input_ids, token_type_ids = qa_enc['input_ids'], qa_enc['token_type_ids']
                start_scores, end_scores = self.model(
                    torch.tensor([input_ids]),
                    token_type_ids=torch.tensor([token_type_ids])
                )

                all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1])

                answers.append(dict(
                    qid=qa['qid'],
                    text=answer,
                    probability=None,
                    start_logits=start_scores,
                    end_logits=end_scores,
                ))

        return answers

    def find_answers(self, data):
        answers = None
        # by now, only batch process on GPU
        if self.params['device'] != 'cpu' and len(data['paragraphs']) > 50:
            # allow default answer len and n_best size
            # ToDo := Add to public API
            answers = self.find_answers_batch(data)
        else:
            answers = self.find_answers_simple(data)
        return answers

    @classmethod
    def from_pretrained(cls, path_or_name, params=None):
        model = BertForQuestionAnswering.from_pretrained(path_or_name)
        tokenizer = BertTokenizer.from_pretrained(path_or_name)
        user_params = default_params.copy()
        if params is not None:
            user_params.update(**params)
        return cls(model, tokenizer, user_params)
