from agent.hyperparameters import PreprocessorHyperparameters, ImageType

from skimage.transform import resize
from skimage.io import imsave

import numpy as np

class ImagePreprocessor(object):

    def __init__(self, hyperparameters=PreprocessorHyperparameters()):
        self.hyperparameters = hyperparameters

    def process(self, rgb_image):
        resized_rgb_image = resize(image=rgb_image.astype(float),
                                   output_shape=(self.hyperparameters.OUTPUT_HEIGHT, self.hyperparameters.OUTPUT_WIDTH),
                                   preserve_range=False)

        if self.hyperparameters.OUTPUT_TYPE == ImageType.RGB:
            return self._normalize(resized_rgb_image)
        elif self.hyperparameters.OUTPUT_TYPE == ImageType.Y:
            return self._normalize(self._luminance_channel(resized_rgb_image))
        elif self.hyperparameters.OUTPUT_TYPE == ImageType.RGBY:
          return self._normalize(self._add_luminance_channel(resized_rgb_image))
        else:
            raise ValueError('Preprocessor Hyperparameter output type not expected')

    def _normalize(self, image):
        return np.float16(image / 255.0)

    def _luminance_channel(self, rgb_image):
        rgb_to_y = np.array([[0.33],
                             [0.33],
                             [0.33]])

        return rgb_image.dot(rgb_to_y)

    def _add_luminance_channel(self, rgb_image):
        rgb_to_rgby = np.array([[1.0, 0.0, 0.0, 0.33],
                                [0.0, 1.0, 0.0, 0.33],
                                [0.0, 0.0, 1.0, 0.33]])

        return rgb_image.dot(rgb_to_rgby)

    def _debug_save_lumincance(self, y_image):
        imsave("debug_yframe_luma.png", np.divide(y_image[:, :, 0].astype(float), 256.0), plugin='pil')

    def _debug_save_rgb_and_lumincance(self, rgby_image):
        imsave("debug_rgbyframe_rgb.png", np.divide(rgby_image[:, :, 0:3].astype(float), 256.0), plugin='pil')
        imsave("debug_rgbyframe_luma.png", np.divide(rgby_image[:, :, 3].astype(float), 256.0), plugin='pil')