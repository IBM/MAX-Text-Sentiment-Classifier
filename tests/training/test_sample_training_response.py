#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import requests


def test_response():

    # test code 200
    model_endpoint = 'http://localhost:5000/model/predict'

    json_data = {
        "text": ["good string",
                 "bad string"]
    }

    r = requests.post(url=model_endpoint, json=json_data)

    assert r.status_code == 200
    response = r.json()
    assert response['status'] == 'ok'

    # test whether the labels have changed
    assert 'pos' in response['predictions'][0].keys()
    assert 'neg' in response['predictions'][0].keys()


if __name__ == '__main__':
    pytest.main([__file__])
