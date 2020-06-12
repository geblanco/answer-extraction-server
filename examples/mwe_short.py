import sys
sys.path.append('../src')

from modeling import TransformerQuestionAnswering, QuestionAnswering

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
model_str = 'distilbert-base-uncased-distilled-squad'
config = AutoConfig.from_pretrained(model_str)
tokenizer = AutoTokenizer.from_pretrained(model_str)
model = AutoModelForQuestionAnswering.from_config(config)

from modeling import default_params
from app import preprocess_input
import json

qa_system = TransformerQuestionAnswering(model, config, tokenizer, default_params)
data = json.load(open('squad/dev-v2.0_formatted.json', 'r'))
data = preprocess_input(data=data)
data_10 = data.copy()
data_10['paragraphs'] = data_10['paragraphs'][:10]
# answers = qa_system.find_answers_simple(data_10)
answers = qa_system.find_answers_batch(data_10)
