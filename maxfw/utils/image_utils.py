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
from __future__ import division
import sys
from PIL import Image
import collections

from . import image_functions as F

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class ImageProcessor(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): sequence of transforms to compose.

    Example:
        >>> pipeline = ImageProcessor([
        >>>     Rotate(150),
        >>>     Resize([100,100])
        >>> ])
        >>> pipeline.apply_transforms(img)
    """

    def __init__(self, transforms=[]):
        assert isinstance(transforms, Sequence)  # nosec - assert
        self.transforms = transforms

    def apply_transforms(self, img):
        """
        Sequentially apply the list of transformations to the input image.

        args:
            img: an image in bytes format, as a Pillow image object, or a numpy ndarray

        output:
            The transformed image.
            Depending on the transformation the output is either a Pillow Image object or a numpy ndarray.
        """
        # verify whether the Normalize or Standardize transformations are positioned at the end
        encoding = [(isinstance(t, Normalize) or isinstance(t, Standardize)) for t in self.transforms]
        if sum(encoding[:-1]) != 0:
            raise ValueError('A Standardize or Normalize transformation can only be positioned at the end of the'
                             'pipeline.')
        # apply the transformations
        for t in self.transforms:
            img = t(img)
        return img


class ToPILImage(object):
    """Convert a byte stream or an ndarray to PIL Image.

    Converts a byte stream or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
             - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
             - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
             - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
             - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
              ``short``).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """
    def __init__(self, target_mode, mode=None):
        self.mode = mode
        self.target_mode = target_mode

    def __call__(self, pic):
        """
        Args:
            pic (bytestream or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        return F.to_pil_image(pic, self.target_mode, self.mode)


class PILtoarray(object):
    """
    Convert a PIL Image object to a numpy ndarray.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image): Image to be converted to a numpy ndarray.

        Returns:
            numpy ndarray
        """
        return F.pil_to_array(pic)


class Normalize(object):
    """
    Normalize the image to a range between [0, 1].
    """

    def __call__(self, img):
        """
        Args:
        img (PIL image or numpy.ndarray): Image to be normalized.

        Returns:
        numpy.ndarray: Normalized image.
        """
        return F.normalize(img)


class Standardize(object):
    """
    Standardize the image (mean-centering and STD of 1).

    Args:
        mean (optional): a single number or an n-dimensional sequence with n equal to the number of image channels
        std (optional): a single number or an n-dimensional sequence with n equal to the number of image channels
    Returns:
        numpy.ndarray: standardized image

    If `mean` or `std` are not provided, the channel-wise values will be calculated for the input image.
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
        img (PIL image or numpy.ndarray): Image to be standardized.

        Returns:
        numpy.ndarray: Standardized image.
        """
        return F.standardize(img, self.mean, self.std)


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Sequence) and len(size) == 2)  # nosec - assert
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation)


class Rotate(object):
    """
    Rotate the input PIL Image by a given angle (counter clockwise).

    Args:
        angle (int or float): Counter clockwise angle to rotate the image by.
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        return F.rotate(img, self.angle)


class Grayscale(object):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1, 3 or 4) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b
        - If num_output_channels == 4 : returned image is 3 channel with r == g == b == a

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return F.to_grayscale(img, num_output_channels=self.num_output_channels)
