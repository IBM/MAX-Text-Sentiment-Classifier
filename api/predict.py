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

from core.model import ModelWrapper
from maxfw.core import MAX_API, PredictAPI
from flask_restplus import fields
from flask import abort

# Set up parser for input data (http://flask-restplus.readthedocs.io/en/stable/parsing.html)
input_parser = MAX_API.model('ModelInput', {
    'text': fields.List(fields.String, required=True,
                        description='List of claims (strings) to be analyzed for either a positive or negative sentiment.')
})

with open('assets/labels.txt', 'r') as f:
    class_labels = [x.strip() for x in f]

# Creating a JSON response model: https://flask-restplus.readthedocs.io/en/stable/marshalling.html#the-api-model-factory
label_prediction = MAX_API.model('LabelPrediction',
                                 {l: fields.Float(required=True, description='Class probability') for l in class_labels})  # noqa - E741

predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'predictions': fields.List(fields.Nested(label_prediction), description='Predicted labels and probabilities')
})


class ModelPredictAPI(PredictAPI):

    model_wrapper = ModelWrapper()

    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        """Make a prediction given input data"""
        result = {'status': 'error'}

        input_json = MAX_API.payload

        try:
            preds = self.model_wrapper.predict(input_json['text'])
        except: # noqa
            abort(400, "Please supply a valid input json. "
                       "The json structure should have a 'text' field containing a list of strings")

        # Generate the output format for every input string
        output = [{l: p[i] for i, l in enumerate(class_labels)} for p in preds]

        result['predictions'] = output
        result['status'] = 'ok'

        return result
