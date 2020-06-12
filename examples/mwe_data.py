import os
import sys
import json
import torch
import concurrent.futures as cf
import timeit

sys.path.append('../src')

from app import preprocess_input
from tqdm import tqdm
from functools import partial
from modeling import default_params
from modeling import TransformerQuestionAnswering, QuestionAnswering
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from multiprocessing import Pool, cpu_count
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertTokenizer,
    BertForQuestionAnswering,
    AutoModelForQuestionAnswering,
)
from transformers.data.processors.squad import (
    SquadExample,
    SquadResult,
    squad_convert_examples_to_features,
    squad_convert_example_to_features,
)

from transformers.data.metrics.squad_metrics import compute_predictions_logits

model_str = 'distilbert-base-uncased-distilled-squad'
config = AutoConfig.from_pretrained(model_str)
tokenizer = AutoTokenizer.from_pretrained(model_str)
model = AutoModelForQuestionAnswering.from_config(config)

qa_system = TransformerQuestionAnswering(model, config, tokenizer, default_params)
data = json.load(open('squad/dev-v2.0_formatted.json', 'r'))
data = preprocess_input(data=data)

data_10 = data.copy()
data_10['paragraphs'] = data_10['paragraphs'][:10]

# answers = qa_system.find_answers_batch(data_10)

start_time = timeit.default_timer()
examples = QuestionAnswering.create_examples(data_10)
features, dataset = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=qa_system.tokenizer,
    max_seq_length=qa_system.params['max_seq_length'],
    doc_stride=qa_system.params['doc_stride'],
    max_query_length=qa_system.params['max_query_length'],
    is_training=False,
    return_dataset="pt",
    threads=8,
)
sampler = SequentialSampler(dataset)
dataloader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=qa_system.params['batch_size']
)
evalTime = timeit.default_timer() - start_time