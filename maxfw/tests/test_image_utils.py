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
# Standard libs
import io

# Dependencies
import nose
import numpy as np
from PIL import Image

# The module to test
from maxfw.utils.image_utils import ImageProcessor, ToPILImage, Resize, Grayscale, Normalize, Standardize, Rotate, \
    PILtoarray
from maxfw.core.utils import MAXImageProcessor

# Initialize a test input file
stream = io.BytesIO()
Image.open('maxfw/tests/test_image.jpg').convert('RGBA').save(stream, 'PNG')
test_input = stream.getvalue()


def test_imageprocessor_read():
    """Test the Imageprocessor."""

    # Test with 4 channels
    transform_sequence = [ToPILImage('RGBA')]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.array(img_out).shape == (678, 1024, 4)

    # Test with 3 channels
    transform_sequence = [ToPILImage('RGB')]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.array(img_out).shape == (678, 1024, 3)

    # Test the values of the image
    transform_sequence = [ToPILImage('RGBA'), PILtoarray()]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.min(img_out) >= 0
    assert np.max(img_out) <= 255

    # Test the values of the image
    transform_sequence = [ToPILImage('L'), PILtoarray()]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.min(img_out) >= 0
    assert np.max(img_out) <= 255


def test_imageprocessor_resize():
    """Test the Imageprocessor's resize function."""

    # Resize to 200x200
    transform_sequence = [ToPILImage('RGBA'), Resize(size=(200, 200))]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.array(img_out).shape == (200, 200, 4)

    # Resize to 2000x2000
    transform_sequence = [ToPILImage('RGBA'), Resize(size=(2000, 2000))]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.array(img_out).shape == (2000, 2000, 4)


def test_imageprocessor_grayscale():
    """Test the Imageprocessor's grayscale function."""

    # Using the standard 1 output channel
    transform_sequence = [ToPILImage('RGBA'), Resize(size=(200, 200)), Grayscale()]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.array(img_out).shape == (200, 200)

    # Using 3 output channels
    transform_sequence = [ToPILImage('RGBA'), Resize(size=(200, 200)), Grayscale(num_output_channels=3)]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.array(img_out).shape == (200, 200, 3)

    # Using 4 output channels
    transform_sequence = [ToPILImage('RGBA'), Resize(size=(200, 200)), Grayscale(num_output_channels=4), PILtoarray()]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert img_out.shape == (200, 200, 4)

    # Test that the values in all 4 output channels are identical
    assert img_out[..., 0].all() == img_out[..., 1].all() == img_out[..., 2].all() == img_out[..., 3].all()


def test_imageprocessor_normalize():
    """Test the Imageprocessor's normalize function."""

    # Test normalize
    transform_sequence = [ToPILImage('RGBA'), Normalize()]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.max(img_out) <= 1 and np.min(img_out) >= 0

    # Test normalize
    transform_sequence = [ToPILImage('RGB'), Normalize()]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.max(img_out) <= 1 and np.min(img_out) >= 0

    # Test normalize
    transform_sequence = [ToPILImage('L'), Normalize()]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.max(img_out) <= 1 and np.min(img_out) >= 0

    # Test for wrong use
    transform_sequence = [ToPILImage('L'), Normalize(), Resize(size=(200, 200))]
    p = ImageProcessor(transform_sequence)
    with nose.tools.assert_raises(Exception):
        p.apply_transforms(test_input)

    # Test for wrong use
    transform_sequence = [ToPILImage('RGBA'), Normalize(), Normalize()]
    p = ImageProcessor(transform_sequence)
    with nose.tools.assert_raises(Exception):
        p.apply_transforms(test_input)


def test_imageprocessor_standardize():
    """Test the Imageprocessor's standardize function."""

    # Test standardize (RGBA)
    # The `A` channel cannot be standardized, and will therefore prompt an error message when attempted.
    transform_sequence = [ToPILImage('RGBA'), Standardize()]
    with nose.tools.assert_raises_regexp(ValueError, r".*must be converted to an image with 3 or fewer channels*"):
        ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize (RGB)
    transform_sequence = [ToPILImage('RGB'), Standardize()]
    p = ImageProcessor(transform_sequence)
    img_out = np.array(p.apply_transforms(test_input))
    nose.tools.assert_almost_equal(np.std(img_out), 1)
    nose.tools.assert_almost_equal(np.mean(img_out), 0)

    # Test standardize (RGB) - check whether the mean centering works
    transform_sequence = [ToPILImage('RGB'), Standardize(std=0)]
    p = ImageProcessor(transform_sequence)
    img_out = np.array(p.apply_transforms(test_input))
    nose.tools.assert_almost_equal(np.mean(img_out), 0)

    # Test standardize (RGB) - check whether the std division works
    transform_sequence = [ToPILImage('RGB'), Standardize(mean=0)]
    p = ImageProcessor(transform_sequence)
    img_out = np.array(p.apply_transforms(test_input))
    nose.tools.assert_almost_equal(np.std(img_out[..., 0]), 1)
    nose.tools.assert_almost_equal(np.std(img_out[..., 1]), 1)
    nose.tools.assert_almost_equal(np.std(img_out[..., 2]), 1)

    # generate an image array
    pil_img = ImageProcessor([ToPILImage('RGB'), PILtoarray()]).apply_transforms(test_input)

    # Test standardize (RGB) with 3 channel-wise values
    transform_sequence = [ToPILImage('RGB'), Standardize(mean=[x for x in np.mean(pil_img, axis=(0, 1))],
                                                         std=[x for x in np.std(pil_img, axis=(0, 1))])]
    ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize (RGB) with 3 channel-wise values for the mean
    transform_sequence = [ToPILImage('RGB'), Standardize(mean=[x for x in np.mean(pil_img, axis=(0, 1))])]
    ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize (RGB) with 3 channel-wise values for the std
    transform_sequence = [ToPILImage('RGB'), Standardize(std=[x for x in np.std(pil_img, axis=(0, 1))])]
    ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize (RGB)
    transform_sequence = [ToPILImage('RGB'), Standardize(mean=np.mean(pil_img))]
    ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize (RGB)
    transform_sequence = [ToPILImage('RGB'), Standardize(mean=np.mean(pil_img), std=np.std(pil_img))]
    ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize (RGB)
    transform_sequence = [ToPILImage('RGB'), Standardize(std=np.std(pil_img))]
    ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize (RGB)
    transform_sequence = [ToPILImage('RGB'), Standardize(mean=127, std=5)]
    ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize error (RGB)
    transform_sequence = [ToPILImage('RGB'), Standardize(mean=[127, 127], std=5)]
    with nose.tools.assert_raises_regexp(ValueError, r".*must correspond to the number of channels.*"):
        ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize error (RGB)
    transform_sequence = [ToPILImage('RGB'), Standardize(std=[5, 5, 5, 5])]
    with nose.tools.assert_raises_regexp(ValueError, r".*must correspond to the number of channels.*"):
        ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize (L)
    transform_sequence = [ToPILImage('L'), Standardize()]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    nose.tools.assert_almost_equal(np.std(img_out), 1)

    # Test standardize (L)
    transform_sequence = [ToPILImage('L'), Standardize(mean=np.mean(pil_img))]
    ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize (L)
    transform_sequence = [ToPILImage('L'), Standardize(mean=np.mean(pil_img), std=np.std(pil_img))]
    ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize (L)
    transform_sequence = [ToPILImage('L'), Standardize(std=np.std(pil_img))]
    ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize (L)
    transform_sequence = [ToPILImage('L'), Standardize(mean=127, std=5)]
    ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize (L) - check whether the mean centering works
    transform_sequence = [ToPILImage('L'), Standardize(std=0)]
    p = ImageProcessor(transform_sequence)
    img_out = np.array(p.apply_transforms(test_input))
    nose.tools.assert_almost_equal(np.mean(img_out), 0)

    # Test standardize (L) - check whether the std division works
    transform_sequence = [ToPILImage('L'), Standardize(mean=0)]
    p = ImageProcessor(transform_sequence)
    img_out = np.array(p.apply_transforms(test_input))
    nose.tools.assert_almost_equal(np.std(img_out), 1)

    # Test standardize error (L)
    transform_sequence = [ToPILImage('L'), Standardize(mean=[127, 127], std=5)]
    with nose.tools.assert_raises(ValueError):
        ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test standardize error (L)
    transform_sequence = [ToPILImage('L'), Standardize(std=[5, 5, 5, 5])]
    with nose.tools.assert_raises(ValueError):
        ImageProcessor(transform_sequence).apply_transforms(test_input)

    # Test for wrong use (L)
    transform_sequence = [ToPILImage('L'), Standardize(), Resize(size=(200, 200))]
    p = ImageProcessor(transform_sequence)
    with nose.tools.assert_raises(Exception):
        p.apply_transforms(test_input)

    # Test for wrong use (RGBA)
    transform_sequence = [ToPILImage('RGBA'), Standardize(), Standardize()]
    p = ImageProcessor(transform_sequence)
    with nose.tools.assert_raises(Exception):
        p.apply_transforms(test_input)


def test_imageprocessor_rotate():
    """Test the Imageprocessor's rotate function."""

    # Test rotate (int)
    transform_sequence = [ToPILImage('RGBA'), Resize((200, 200)), Rotate(5)]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.array(img_out).shape == (200, 200, 4)

    # Test rotate vs Pillow rotate
    transform_sequence = [ToPILImage('RGBA'), Resize((200, 200))]
    p = ImageProcessor(transform_sequence)
    img_in = p.apply_transforms(test_input)

    transform_sequence = [Rotate(5)]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(img_in)
    assert img_in.rotate(5) == img_out

    # Test rotate (negative int)
    transform_sequence = [ToPILImage('RGBA'), Resize((200, 200)), Rotate(-5)]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.array(img_out).shape == (200, 200, 4)

    # Test rotate (float)
    transform_sequence = [ToPILImage('RGBA'), Resize((200, 200)), Rotate(499.99)]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.array(img_out).shape == (200, 200, 4)


def test_imageprocessor_combinations():
    """Test various combinations of the Imageprocessor's functionality."""

    # Combination 1
    transform_sequence = [
        ToPILImage('RGB'),
        Resize((2000, 2000)),
        Rotate(5),
        Grayscale(num_output_channels=4),
        Resize((200, 200)),
        Normalize()
    ]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.array(img_out).shape == (200, 200, 4)

    # Combination 2
    transform_sequence = [
        ToPILImage('RGB'),
        Resize((200, 200)),
        Normalize()
    ]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.array(img_out).shape == (200, 200, 3)

    # Combination 3
    transform_sequence = [
        ToPILImage('RGB'),
        Resize((200, 200)),
        Standardize()
    ]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert np.array(img_out).shape == (200, 200, 3)

    # Combination 5 - including pixel value tests
    transform_sequence = [
        ToPILImage('RGB'),
        Resize((2000, 2000)),
        Rotate(5),
        Grayscale(num_output_channels=4),
        Resize((200, 200)),
        PILtoarray()
    ]
    p = ImageProcessor(transform_sequence)
    img_out = p.apply_transforms(test_input)
    assert img_out.shape == (200, 200, 4)
    assert np.min(img_out) >= 0
    assert np.max(img_out) <= 255

    # Combination 6 - utilize the pipeline multiple times
    transform_sequence = [
        ToPILImage('RGB'),
        Resize((2000, 2000)),
        Rotate(5),
        Grayscale(num_output_channels=4),
        Resize((200, 200)),
        PILtoarray()
    ]
    p = ImageProcessor(transform_sequence)
    p.apply_transforms(test_input)
    p.apply_transforms(test_input)


def test_flask_error():

    # Test invalid input format
    transform_sequence = [ToPILImage('RGB')]
    p = MAXImageProcessor(transform_sequence)
    with nose.tools.assert_raises_regexp(Exception, r".*Invalid input format*"):
        p.apply_transforms("")

    # Test invalid input dimensions
    transform_sequence = [ToPILImage('RGB')]
    p = MAXImageProcessor(transform_sequence)
    with nose.tools.assert_raises_regexp(Exception, r".*grayscale or a colour image*"):
        p.apply_transforms(np.random.rand(10, 10, 10, 10))


if __name__ == '__main__':
    nose.main()
