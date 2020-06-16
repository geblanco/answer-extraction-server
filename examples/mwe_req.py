import requests
import base64

def extract_answer(question, context, q_id, c_id):
    """
    Call the web service by UNED and extract an answer in a given context
    :param question:
    :param context:
    :param q_id:
    :param c_id:
    :return:
    """
    encoded_q = base64.b64encode(question.encode('utf8'))
    encoded_c = base64.b64encode(context.encode('utf8'))

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
