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
import os
from flask import Flask
from flask_restx import Api, Namespace
from flask_cors import CORS
from .default_config import API_TITLE, API_DESC, API_VERSION

MAX_API = Namespace('model', description='Model information and inference operations')


class MAXApp(object):

    def __init__(self, title=API_TITLE, desc=API_DESC, version=API_VERSION):
        self.app = Flask(title, static_url_path='')

        # load config
        if os.path.exists("config.py"):
            self.app.config.from_object("config")

        self.api = Api(
            self.app,
            title=title,
            description=desc,
            version=version)

        self.api.namespaces.clear()
        self.api.add_namespace(MAX_API)

        # enable cors if flag is set
        if os.getenv('CORS_ENABLE') == 'true' and \
                (os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or self.app.debug is not True):
            CORS(self.app, origins='*')
            print('NOTE: MAX Model Server is currently allowing cross-origin requests - (CORS ENABLED)')

    def add_api(self, api, route):
        MAX_API.add_resource(api, route)

    def mount_static(self, route):
        @self.app.route(route)
        def index():
            return self.app.send_static_file('index.html')

    def run(self, host='0.0.0.0', port=5000):  # nosec - binding to all interfaces
        self.app.run(host, port)
