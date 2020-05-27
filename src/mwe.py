import torch
import timeit
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    squad_convert_examples_to_features,
)
from transformers.data.processors.squad import SquadV2Processor

data_dir = '/home/gb/Documents/Research/LIHLITH/squad-experiments/squad/'
# dev_file = 'dev-v2.0_correct_format.json'

proc = SquadV2Processor()
examples = proc.get_dev_examples(data_dir)
max_seq_length = 384
pad_on_left = False
pad_token_segment_id = 0
doc_stride = 128
max_query_length = 64
batch_size = 10

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')


features, dataset = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=doc_stride,
    max_query_length=max_query_length,
    is_training=False,
    return_dataset="pt",
    threads=4,
)

sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

times = []
n_runs = 10
run = 0

for batch in tqdm(dataloader, desc="Evaluating"):
    start_time = timeit.default_timer()
    model.eval()
    batch = tuple(t.to('cpu') for t in batch)

    with torch.no_grad():
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }
        outputs = model(**inputs)

    end_time = timeit.default_timer() - start_time
    times.append(end_time)
    run += 1
    if run >= n_runs:
        break

eval_time = sum(times)
print(
    'Processed {} samples in {} secs ({} sec per example)'
    .format(n_runs, eval_time, eval_time / n_runs)
)
