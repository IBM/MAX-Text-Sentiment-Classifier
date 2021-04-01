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

from abc import ABC, abstractmethod


class MAXModelWrapper(ABC):
    def __init__(self, path=None):
        """Implement code to load model here"""
        pass

    def _pre_process(self, x):
        """Implement code to process raw input into format required for model inference here"""
        return x

    def _post_process(self, x):
        """Implement any code to post-process model inference response here"""
        return x

    @abstractmethod
    def _predict(self, x):
        """Implement core model inference code here"""
        pass

    def predict(self, x):
        pre_x = self._pre_process(x)
        prediction = self._predict(pre_x)
        result = self._post_process(prediction)
        return result
