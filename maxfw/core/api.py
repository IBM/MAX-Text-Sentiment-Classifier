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
from .app import MAX_API
from flask_restx import Resource, fields

METADATA_SCHEMA = MAX_API.model('ModelMetadata', {
        'id': fields.String(required=True, description='Model identifier'),
        'name': fields.String(required=True, description='Model name'),
        'description': fields.String(required=True, description='Model description'),
        'type': fields.String(required=True, description='Model type'),
        'source': fields.String(required=True, description='Model source'),
        'license': fields.String(required=False, description='Model license')
    })


class MAXAPI(Resource):
    pass


class MetadataAPI(MAXAPI):

    def get(self):
        """To be implemented"""
        raise NotImplementedError()


class PredictAPI(MAXAPI):

    def post(self):
        """To be implemented"""
        raise NotImplementedError()


class CustomMAXAPI(MAXAPI):
    pass

# class FileRequestParser(object):
#     def __init__(self):
#         self.parser = reqparse.RequestParser()
#         self.parser.add_argument('file', type=FileStorage, location='files', required=True)
