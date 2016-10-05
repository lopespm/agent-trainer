import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import offsetbox
from sklearn import manifold

from agent.trainer.visualization import style


class Tsne(object):

    MIN_DISTANCE_BETWEEN_IMAGES = 0.04

    def __init__(self,
                 number_of_images_per_input,
                 single_input_image_width_input,
                 output_descriptor_enum):
        self.number_of_images_per_input = number_of_images_per_input
        self.single_input_image_width_input = single_input_image_width_input
        self.output_descriptor_enum = output_descriptor_enum

    # TODO
    # This method would not exist ideally. Its only purpose is for matplotlib's backend to have a proper SDL context
    # The game emulator uses SDL as well, if matplotlib is not invoked before the emulator runs, SDL setup fails when plotting
    def init(self):
        plt.figure()

    def save_visualization_to_image(self, inputs, outputs, folder_path_for_result_image):
        print("Computing t-SNE embedding")
        x = np.array([state.reshape(-1, ) for state in inputs])
        y = outputs
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        x_tsne = tsne.fit_transform(x)
        self._tsne_plot_embedding(x=x_tsne,
                                  y=y,
                                  inputs=inputs,
                                  path_result_image=os.path.join(folder_path_for_result_image, "t-SNE.png"))

    def _tsne_plot_embedding(self, x, y, inputs, path_result_image, title=""):
        x_min, x_max = np.min(x, 0), np.max(x, 0)
        x_normalized = (x - x_min) / (x_max - x_min)

        tableau20 = style.generate_tableau20_colors()
        figure = plt.figure()
        figure.set_size_inches(18.5, 10.5)
        ax = figure.add_subplot(111)
        ax.axis('off')
        for i in xrange(x.shape[0]):
            plt.text(x_normalized[i, 0], x_normalized[i, 1], str(y[i]),
                     color=tableau20[y[i]],
                     fontdict={'weight': 'bold', 'size': 12})

        labels = [mpatches.Patch(color=tableau20[output_descriptor.value],
                                 label="[{0}] {1}".format(output_descriptor.value, output_descriptor.name)) for output_descriptor in list(self.output_descriptor_enum)]
        legend = ax.legend(handles=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False)

        if hasattr(offsetbox, 'AnnotationBbox'): # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])
            for i in xrange(len(x_normalized)):
                distance_between_points = np.sum((x_normalized[i] - shown_images) ** 2, 1)
                if np.min(distance_between_points) < self.MIN_DISTANCE_BETWEEN_IMAGES:
                    continue
                shown_images = np.r_[shown_images, [x_normalized[i]]]
                rendered_image = offsetbox.OffsetImage(self._state_into_grid_of_screenshots(inputs[i]),
                                                       cmap=plt.get_cmap('gray'))
                image_position = x_normalized[i]
                annotation_box_relative_position = (-70, 250) if x_normalized[i][1] > 0.5 else (-70, -250)
                imagebox = offsetbox.AnnotationBbox(rendered_image, image_position,
                                                    xybox=annotation_box_relative_position,
                                                    xycoords='data',
                                                    boxcoords="offset points",
                                                    arrowprops=dict(arrowstyle="->"))
                ax.add_artist(imagebox)

        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

        plt.savefig(path_result_image, bbox_extra_artists=(legend,), bbox_inches='tight', pad_inches=4)
        print("Visualization written to {0}".format(path_result_image))

    def _state_into_grid_of_screenshots(self, state):
        grid = state.reshape(self.single_input_image_width_input, -1, order='F')
        for screenshot_index in xrange(1, self.number_of_images_per_input):
            divider_position = (self.single_input_image_width_input * screenshot_index) + screenshot_index
            grid = np.insert(grid, divider_position, 0, axis=1)

        return grid
