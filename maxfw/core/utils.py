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
from flask import abort
from maxfw.utils.image_utils import ImageProcessor


def redirect_errors_to_flask(func):
    """
    This decorator function will capture all Pythonic errors and return them as flask errors.

    If you are looking to disable this functionality, please remove this decorator from the `apply_transforms()` module
    under the ImageProcessor class.
    """

    def inner(*args, **kwargs):
        try:
            # run the function
            return func(*args, **kwargs)
        except ValueError as ve:
            if 'pic should be 2 or 3 dimensional' in str(ve):
                abort(400, "Invalid input, please ensure the input is either "
                      "a grayscale or a colour image.")
        except TypeError as te:
            if 'bytes or ndarray' in str(te):
                abort(400, "Invalid input format, please make sure the input file format "
                      " is a common image format such as JPG or PNG.")
    return inner


class MAXImageProcessor(ImageProcessor):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> pipeline = ImageProcessor([
        >>>     Rotate(150),
        >>>     Resize([100,100])
        >>> ])
        >>> pipeline.apply_transforms(img)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @redirect_errors_to_flask
    def apply_transforms(self, img):
        return super().apply_transforms(img)
