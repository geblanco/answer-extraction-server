import os
import json
import base64
import argparse


from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer


DEFAULT_ROUTE = 'qa-extraction'
DEFAULT_PORT = 8000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', '-m', required=True, type=str, help='Model to serve'
    )
    parse.add_argument(
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


def is_opt_encoded(data):
    return data.get('options', {}).get('encoded', False)


def preprocess_input(data):
    if data is None:
        return None

    if not is_opt_encoded(data):
        return data

    proc_data = data.copy()
    for i, par in enumerate(proc_data['paragraphs']):
        proc_par = par.copy()
        proc_par['context'] = decode_text(par['context'])
        proc_par['qas'] = [
            {
                'qid': qa['qid'],
                'question': decode_text(qa['question'])
            }
            for qa in par['qas']
        ]
        proc_data['paragraphs'][i] = proc_par
    return proc_data


def get_response(model_answers, orig_data):
    if is_opt_encoded(orig_data):
        # encode data
        pass
    return {}


def setup_route(app, route, port, model_path):
    model = Predictor.from_path(model)

    @app.route(f'/{route}', methods=['POST'])
    def serve_route():
        data = preprocess_input(request.get_json())
        if data is None:
            return jsonify({})
        return jsonify(get_response(model.find_answer(data), data))

def serve(model_path, route, port):
    app = Flask(__name__)
    model_path = os.path.join(os.path.abspath(os.path.curdir), model_path)
    print(f'Serving {model_path} in {route}:{port}')
    setup_route(app, route, port, model_path)
    # serve on all interfaces with ip on given port
    http_server = WSGIServer(('0.0.0.0', port), app)
    print(f'Ready for requests!')
    http_server.serve_forever()


if __name__ == '__main__':
    flags = parse_args()
    print('********* Question Answering Server *********')
    print(f'Input flags {flags}')
    serve(flags.model, flags.route, flags.port)
