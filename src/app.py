import os
import base64
import argparse


from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer

# from qa_system import QA_System
from modeling import TransformerQuestionAnswering

DEFAULT_ROUTE = 'qa-extraction'
DEFAULT_PORT = 8000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', '-m', required=True, type=str, help='Model to serve'
    )
    parser.add_argument(
        '--route', '-r', required=False, default=DEFAULT_ROUTE,
        help=f'Route to serve (default: {DEFAULT_ROUTE})'
    )
    parser.add_argument(
        '--port', '-p', required=False, default=DEFAULT_PORT,
        type=int, help='Port to serve'
    )
    return parser.parse_args()


def encode_text(text):
    base64.b64encode(text.encode('utf8'))


def decode_text(text):
    return base64.b64decode(text).decode('utf8')


def add_id_field(example):
    qas = example['qas']
    for i, q_dict in enumerate(qas):
        if q_dict.get('id') is not None:
            continue
        qid = q_dict.get('qid', i)
        q_dict['id'] = qid
    return example['context'], qas


def is_opt_encoded(data):
    return data.get('options', {}).get('encoded', False)


def apply_fn_to_qa_example(data, fn):
    proc_data = data.copy()
    for i, par in enumerate(proc_data['paragraphs']):
        proc_par = par.copy()
        proc_par['context'], proc_par['qas'] = fn(proc_par)
        proc_data['paragraphs'][i] = proc_par
    return proc_data


def apply_fn_to_qa_fields(data, fn):
    proc_data = data.copy()
    for i, par in enumerate(proc_data['paragraphs']):
        proc_par = par.copy()
        proc_par['context'] = fn(par['context'])
        proc_par['qas'] = [
            {
                'qid': qa['qid'],
                'question': fn(qa['question'])
            }
            for qa in par['qas']
        ]
        proc_data['paragraphs'][i] = proc_par
    return proc_data


def format_answers(answers, nbest):
    output_qas = []
    for ans, best in zip(answers.items(), nbest.items()):
        key = ans[0]
        assert(key == best[0])
        output_qas.append(dict(
            qid=key,
            text=ans[1],
            results=best[1]
        ))
    return output_qas


def preprocess_input(data):
    if data is None:
        return None

    if not is_opt_encoded(data):
        return data

    data = apply_fn_to_qa_fields(data, decode_text)
    data = apply_fn_to_qa_example(data, add_id_field)
    return data


def prepare_response(data):
    if is_opt_encoded(data):
        # encode data
        data = apply_fn_to_qa_fields(data, encode_text)
    return data


def setup_route(app, route, port, model_path):
    model = TransformerQuestionAnswering.from_pretrained(model_path)

    @app.route(f'/{route}', methods=['POST'])
    def serve_route():
        # decode if needed
        data = preprocess_input(request.get_json())
        if data is None:
            return jsonify({})
        n_best_size = int(data.get('k', 4))
        answers, nbest = model.find_answers(data, n_best_size)
        output_qas = format_answers(answers, nbest)
        output_data = dict(options=data['options'], qas=output_qas)
        return jsonify(prepare_response(output_data))


def serve(model_path, route, port):
    app = Flask(__name__)
    model_path = os.path.join(os.path.abspath(os.path.curdir), model_path)
    print(f'Serving {model_path} in {route}:{port}')
    setup_route(app, route, port, model_path)
    # serve on all interfaces with ip on given port
    http_server = WSGIServer(('0.0.0.0', port), app)
    print('Ready for requests!')
    http_server.serve_forever()


if __name__ == '__main__':
    flags = parse_args()
    print('********* Question Answering Server *********')
    print(f'Input flags {flags}')
    serve(flags.model, flags.route, flags.port)
