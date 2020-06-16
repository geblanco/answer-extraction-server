import requests
import base64
import json

def extract_answer(question, context, q_id, c_id):
    """
    Call the web service by UNED and extract an answer in a given context
    :param question:
    :param context:
    :param q_id:
    :param c_id:
    :return:
    """
    encoded_q = base64.b64encode(question.encode('utf8')).decode('utf-8')
    encoded_c = base64.b64encode(context.encode('utf8')).decode('utf-8')

    url = "<server>"
    headers = {'Content-Type': 'application/json',
               'charset': 'utf-8'}
    data = {"paragraphs": [{
        "qas": [{
            "qid": q_id,
            "question": encoded_q
        }],
        "context": encoded_c,
        "cid": c_id,
    }],
        "options": {
            "encoded": True
        }
    }
    print(data)
    # using params instead of data because we are making this POST request by
    # constructing query string URL with key/value pairs in it.

    r = requests.post(url, json=data, headers=headers)
    return r

def decode_text(text):
    return base64.b64decode(text).decode('utf8')

def apply_fn_to_qa_fields_answer(data, fn):
    if data is None or data.get('qas') is None:
        return None
    for qa_index, qa in enumerate(data['qas']):
        qa['text'] = fn(qa['text'])
        for i, result in enumerate(qa['results']):
            result['text'] = fn(result['text'])
            qa['results'][i] = result
        data['qas'][qa_index] = qa
    return data


context = """One of the first Norman mercenaries to serve as a Byzantine general
was Herv in the 1050s. By then however, there were already Norman
mercenaries serving as far away as Trebizond and Georgia. They were based at
Malatya and Edessa, under the Byzantine duke of Antioch, Isaac Komnenos. In the
1060s, Robert Crispin led the Normans of Edessa against the Turks. Roussel de
Bailleul even tried to carve out an independent state in Asia Minor with
support from the local population, but he was stopped by the Byzantine general
Alexius Komnenos."""

question = "What was the name of the Norman castle?"
qid = '56de10b44396321400ee2593'
cid = '56de10b44396321400ee2594'

response = extract_answer(question, context, qid, cid)
print(json.dumps(apply_fn_to_qa_fields_answer(response.json(), decode_text), indent=2))
