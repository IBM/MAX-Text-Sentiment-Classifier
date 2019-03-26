import pytest
import requests


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'MAX Text Sentiment Classifier'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'sentiment_bert_base_uncased_tensorflow'
    assert metadata['name'] == 'Bert Base Uncased TensorFlow Model'
    assert metadata['description'] == 'BERT Base finetuned on the IBM Project Debater Claim Sentiment dataset.'
    assert metadata['license'] == 'ApacheV2'
    assert metadata['type'] == 'Text Classification'
    assert 'developer.ibm.com' in metadata['source']


def test_response():
    model_endpoint = 'http://localhost:5000/model/predict'

    json_data = {
        "text": ["good string",
                 "bad string"]
    }

    r = requests.post(url=model_endpoint, json=json_data)

    assert r.status_code == 200
    response = r.json()
    assert response['status'] == 'ok'

    # verify that 'good string' is in fact positive
    print(response['predictions'][0])
    assert round(float(response['predictions'][0][0]['positive'])) == 1
    # verify that 'bad string' is in fact negative
    assert round(float(response['predictions'][1][0]['negative'])) == 1

    json_data2 = {
        "text": [
            "2008 was a dark, dark year for stock markets worldwide.",
            "The Model Asset Exchange is a crucial element of a developer's toolkit."
        ]
    }

    r = requests.post(url=model_endpoint, json=json_data2)

    assert r.status_code == 200
    response = r.json()
    assert response['status'] == 'ok'

    # verify that "2008 was a dark, dark year for stock markets worldwide." is in fact negative
    assert round(float(response['predictions'][0][0]['positive'])) == 0
    assert round(float(response['predictions'][0][0]['negative'])) == 1
    # verify that "The Model Asset Exchange is a crucial element of a developer's toolkit." is in fact positive
    assert round(float(response['predictions'][1][0]['negative'])) == 0
    assert round(float(response['predictions'][1][0]['positive'])) == 1

    # Test different input batch sizes
    for input_size in [4, 16, 32, 64, 75]:
        json_data3 = {
            "text": ["good string"]*input_size
        }

        r = requests.post(url=model_endpoint, json=json_data3)

        assert r.status_code == 200
        response = r.json()
        assert response['status'] == 'ok'

        assert len(response['predictions']) == len(json_data3["text"])


if __name__ == '__main__':
    pytest.main([__file__])
