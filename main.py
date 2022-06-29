import requests
from requests.auth import HTTPBasicAuth
import json
import re


def remove_emojis(data):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def get_cleared_sentences(data):
    return [remove_emojis(d) for d in data]


def get_data(test=False, size=10_000):
    filter_query = dict()
    if test:
        filter_query['bool'] = {
            'must': [{
                'match': {
                    'lang': "pl",
                },
            },
                {
                'match': {
                    'is_retweet': False,
                },
            },
            ],
            'must_not': [{
                'match': {
                    'is_hate_speech': True,
                },
            },
                {
                'match': {
                    'is_hate_speech': False,
                },
            }, ],
        }
    else:
        filter_query['bool'] = {
            'must': [{
                'match': {
                    'lang': "pl",
                },
            },
                {
                'match': {
                    'is_retweet': False,
                },
            },
            ],
            'should': [{
                'match': {
                    'is_hate_speech': True,
                },
            },
                {
                'match': {
                    'is_hate_speech': False,
                },
            }, ],
        }

    url = 'https://es.dc9.dev:9200/tweets/_search'
    headers = {'content-type': 'application/json'}
    query = {
        'size': size,
        "_source": ["content", "is_hate_speech"],
        'query': {
            'function_score': {
                'random_score': {},
                'query': filter_query
            },
        },
    }

    r = requests.get(url=url, data=json.dumps(query),
                     headers=headers, auth=HTTPBasicAuth('dc9', 'hohC2wix'))
    return r.json()["hits"]["hits"]


def get_sentences_from_data(data):
    return [d["_source"]["content"] for d in data]


def main():
    data = get_data(test=True, size=10)
    raw_sentences = get_sentences_from_data(data)
    print(raw_sentences)
    sentences = get_cleared_sentences(raw_sentences)
    print(sentences)


if __name__ == '__main__':
    main()
