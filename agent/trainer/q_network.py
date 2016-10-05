from agent.hyperparameters import QNetworkHyperparameters

import numpy as np
import tensorflow as tf
import math
import logging

from collections import namedtuple


TFGraphForwardPassBundle = namedtuple('TFGraphForwardPropBundle',
                                      ['input_state',
                                       'output_all_actions_q_values',
                                       'variable_scope_name_prefix'])

TFGraphTrainBundle = namedtuple('TFGraphTrainBundle',
                                ['input_states',
                                 'output_all_actions_q_values',
                                 'action_indexes',
                                 'target_action_q_values',
                                 'learning_rate',
                                 'loss',
                                 'optimizer',
                                 'variable_scope_name_prefix'])

QNetworkTrainBundle = namedtuple("QNetworkTrainBundle", ["state", "action_index", "target_action_q_value"])

class QNetworkFactory(object):
    def create(self, screen_width, screen_height, num_channels, num_actions, metrics_directory, batched_forward_pass_size):
        return QNetwork(screen_width, screen_height, num_channels, num_actions, metrics_directory, batched_forward_pass_size)

class QNetwork(object):

    MODEL_NAME_TRAIN = 'model-train'
    MODEL_NAME_FORWARD_PASS = 'model-forward-pass'

    def __init__(self,
                 screen_width,
                 screen_height,
                 num_channels,
                 num_actions,
                 metrics_directory,
                 batched_forward_pass_size,
                 hyperparameters=QNetworkHyperparameters()):
        self.logger = logging.getLogger(__name__)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.batched_forward_pass_size = batched_forward_pass_size
        self.hyperparameters = hyperparameters

        self.tf_graph = tf.Graph()
        self.tf_graph_forward_pass_bundle_single = self._build_graph_forward_pass_bundle(self.tf_graph, 1)
        self.tf_graph_forward_pass_bundle_batched = self._build_graph_forward_pass_bundle(self.tf_graph, batched_forward_pass_size)
        self.tf_graph_train_bundle = self._build_graph_train_bundle(self.tf_graph)

        self.tf_session = tf.Session(graph=self.tf_graph)

        with self.tf_graph.as_default():
            self.tf_all_summaries = tf.merge_all_summaries()
            self.tf_summary_writer = tf.train.SummaryWriter(logdir=metrics_directory, graph=self.tf_graph)
            self.tf_saver = tf.train.Saver()
            tf.initialize_all_variables().run(session=self.tf_session)

        self.assigns_train_to_forward_pass_variables = self._build_assigns_train_to_forward_pass_variables()


    def _build_graph_forward_pass_bundle(self, graph, batch_size):
        with graph.as_default():
            input_state = tf.placeholder(tf.float32,
                                         shape=(batch_size, self.screen_height, self.screen_width, self.num_channels),
                                         name='input_state')

            variable_scope_name_prefix = "{0}-{1}-scope".format(self.MODEL_NAME_FORWARD_PASS, batch_size)
            output_all_actions_q_values = self._network_model(variable_scope_name_prefix=variable_scope_name_prefix,
                                                              input=input_state,
                                                              output_size=self.num_actions,
                                                              record_metrics=False)
            return TFGraphForwardPassBundle(input_state=input_state,
                                            output_all_actions_q_values=output_all_actions_q_values,
                                            variable_scope_name_prefix=variable_scope_name_prefix)

    def _build_graph_train_bundle(self, graph):
        with graph.as_default():

            input_states = tf.placeholder(tf.float32,
                                          shape=(self.hyperparameters.SGD_BATCH_SIZE, self.screen_height, self.screen_width, self.num_channels),
                                          name='input_states')

            variable_scope_name_prefix=self.MODEL_NAME_TRAIN
            output_all_actions_q_values = self._network_model(variable_scope_name_prefix=variable_scope_name_prefix,
                                                              input=input_states,
                                                              output_size=self.num_actions,
                                                              record_metrics=True)

            action_indexes = tf.placeholder(tf.float32, shape=(self.hyperparameters.SGD_BATCH_SIZE, self.num_actions), name='action_indexes')
            output_filtered_action_q_values = tf.reduce_sum(tf.mul(output_all_actions_q_values, action_indexes), reduction_indices=1)

            target_action_q_values = tf.placeholder(tf.float32, shape=(self.hyperparameters.SGD_BATCH_SIZE), name='target_action_q_values')
            delta = target_action_q_values - output_filtered_action_q_values
            loss = tf.reduce_mean(tf.square(delta))
            learning_rate = tf.Variable(self.hyperparameters.LEARNING_RATE_INITIAL, trainable=False)
            optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                                  decay=self.hyperparameters.RMS_DECAY,
                                                  momentum=self.hyperparameters.RMS_MOMENTUM,
                                                  epsilon=self.hyperparameters.RMS_EPSILON).minimize(loss)

            tf.scalar_summary('loss', loss)
            tf.scalar_summary('learning_rate', learning_rate)

        return TFGraphTrainBundle(input_states=input_states,
                                  output_all_actions_q_values=output_all_actions_q_values,
                                  action_indexes=action_indexes,
                                  target_action_q_values=target_action_q_values,
                                  learning_rate=learning_rate,
                                  loss=loss,
                                  optimizer=optimizer,
                                  variable_scope_name_prefix=variable_scope_name_prefix)

    def _network_model(self, variable_scope_name_prefix, input, output_size, record_metrics):
        conv1 = self._convolutional_layer(input=input,
                                          patch_size=8,
                                          stride=4,
                                          input_channels=self.num_channels,
                                          output_channels=32,
                                          bias_init_value=0.0,
                                          scope_name=variable_scope_name_prefix + '_conv1')

        conv2 = self._convolutional_layer(input=conv1,
                                          patch_size=4,
                                          stride=2,
                                          input_channels=32,
                                          output_channels=64,
                                          bias_init_value=0.1,
                                          scope_name=variable_scope_name_prefix + '_conv2')

        conv3 = self._convolutional_layer(input=conv2,
                                          patch_size=3,
                                          stride=1,
                                          input_channels=64,
                                          output_channels=64,
                                          bias_init_value=0.1,
                                          scope_name=variable_scope_name_prefix + '_conv3')

        flattened_conv3 = tf.reshape(conv3, [input.get_shape()[0].value, -1])
        flattened_conv3_size = flattened_conv3.get_shape()[1].value

        # relu4
        relu4 = self._relu_layer(input=flattened_conv3,
                                 input_size=flattened_conv3_size,
                                 output_size=512,
                                 scope_name=variable_scope_name_prefix + '_relu4')

        local5 = self._linear_layer(input=relu4,
                                    input_size=512,
                                    output_size=output_size,
                                    scope_name=variable_scope_name_prefix + '_local5')

        if record_metrics:
            self._activation_summary(conv1)
            self._activation_summary(conv2)
            self._activation_summary(conv3)
            self._activation_summary(relu4)
            self._activation_summary(local5)

        return local5

    def _convolutional_layer(self, input, patch_size, stride, input_channels, output_channels, bias_init_value, scope_name):
        with tf.variable_scope(scope_name) as scope:
            weights = tf.get_variable(name='weights',
                                  shape=[patch_size, patch_size, input_channels, output_channels],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
            biases = tf.Variable(name='biases', initial_value=tf.constant(value=bias_init_value, shape=[output_channels]))
            conv = tf.nn.conv2d(input, weights, [1, stride, stride, 1], padding='SAME')

            linear_rectification_bias = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(linear_rectification_bias, name=scope.name)

            grid_x = output_channels // 4
            grid_y = 4 * input_channels
            kernels_image_grid = self._create_kernels_image_grid(weights, (grid_x, grid_y))
            tf.image_summary(scope_name + '/features', kernels_image_grid, max_images=1)

            if "_conv1" in scope_name:
                x_min = tf.reduce_min(weights)
                x_max = tf.reduce_max(weights)
                weights_0_to_1 = (weights - x_min) / (x_max - x_min)
                weights_0_to_255_uint8 = tf.image.convert_image_dtype(weights_0_to_1, dtype=tf.uint8)

                # to tf.image_summary format [batch_size, height, width, channels]
                weights_transposed = tf.transpose(weights_0_to_255_uint8, [3, 0, 1, 2])

                tf.image_summary(scope_name + '/features', weights_transposed[:,:,:,0:1], max_images=32)

        return output

    def _relu_layer(self, input, input_size, output_size, scope_name):
        with tf.variable_scope(scope_name) as scope:
            weights = tf.get_variable(name='weights',
                                      shape=[input_size, output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.Variable(name='biases', initial_value=tf.constant(value=0.1, shape=[output_size]))
            output = tf.nn.relu(tf.matmul(input, weights) + biases, name=scope.name)
        return output

    def _linear_layer(self, input, input_size, output_size, scope_name):
        with tf.variable_scope(scope_name) as scope:
            weights = tf.Variable(name='weights',
                                  initial_value=tf.truncated_normal(shape=[input_size, output_size], stddev=0.1))
            biases = tf.Variable(name='biases', initial_value=tf.constant(value=0.1, shape=[output_size]))
            output = tf.matmul(input, weights) + biases
        return output

    def _activation_summary(self, tensor):
        tensor_name = tensor.op.name
        tf.histogram_summary(tensor_name + '/activations', tensor)
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor))

    def _create_kernels_image_grid(self, kernel, (grid_X, grid_Y), pad=1):
        '''Visualize conv. features as an image (mostly for the 1st layer).
        Place kernel into a grid, with some paddings between adjacent filters.
        Args:
          kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
          (grid_X, grid_Y):  shape of the grid. Require: NumKernels == grid_X * grid_Y
                               User is responsible of how to break into two multiples.
          pad:               number of black pixels around each filter (between them)

        Return:
          Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
        '''

        flattened_kernel = tf.reshape(kernel, tf.pack([kernel.get_shape()[0],
                                                       kernel.get_shape()[1],
                                                       1,
                                                       kernel.get_shape()[3] * kernel.get_shape()[2]]))

        # X and Y dimensions, w.r.t. padding
        Y = flattened_kernel.get_shape()[0] + pad
        X = flattened_kernel.get_shape()[1] + pad

        # pad X and Y
        x1 = tf.pad(flattened_kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 1]))

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 1]))

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scale to [0, 1]
        x_min = tf.reduce_min(x7)
        x_max = tf.reduce_max(x7)
        x8 = (x7 - x_min) / (x_max - x_min)

        return x8

    def create_train_bundle(self, state, action_index, target_action_q_value):
        return QNetworkTrainBundle(state=state,
                                   action_index=action_index,
                                   target_action_q_value=target_action_q_value)

    def train(self, train_bundles, training_step):

        train_bundles_len = len(train_bundles)

        if train_bundles_len < self.hyperparameters.SGD_BATCH_SIZE:
            self.logger.info('Training bundle is smaller than batch size, skipping train')
            return

        offset = self.hyperparameters.SGD_BATCH_SIZE % train_bundles_len
        batch_train_bundles = self._circular_selection_of_batch(offset, train_bundles, train_bundles_len)
        batch_input_states = [train_bundle.state for train_bundle in batch_train_bundles]
        batch_action_indexes = [np.eye(self.num_actions)[train_bundle.action_index] for train_bundle in
                                batch_train_bundles]
        batch_target_action_q_values = [train_bundle.target_action_q_value for train_bundle in batch_train_bundles]

        feed_dict = {
            self.tf_graph_train_bundle.input_states: np.asarray(batch_input_states),
            self.tf_graph_train_bundle.action_indexes: np.asarray(batch_action_indexes),
            self.tf_graph_train_bundle.target_action_q_values: np.asarray(batch_target_action_q_values)
        }

        with self.tf_session.as_default():

            run_result = self.tf_session.run(
                [self.tf_graph_train_bundle.loss,
                 self.tf_graph_train_bundle.optimizer],
                feed_dict=feed_dict)

            evaluated_loss = run_result[0]
            self.logger.info('Loss: %f' % evaluated_loss)

            if training_step % self.hyperparameters.NUM_STEPS_ASSIGN_TRAIN_TO_FORWARD_PROP_GRAPH == 0:
                self.tf_session.run(self.assigns_train_to_forward_pass_variables)
                self.logger.info("Assigning trained variables to forward pass graph")

            if (training_step + 1) % self.hyperparameters.LEARNING_RATE_DECAY_STEP == 0:
                current_learning_rate = self.tf_session.run([self.tf_graph_train_bundle.learning_rate])[0]
                learning_rate_decay = math.pow(float(self.hyperparameters.LEARNING_RATE_FINAL) / float(self.hyperparameters.LEARNING_RATE_INITIAL),
                                               1.0 / (float(self.hyperparameters.LEARNING_RATE_FINAL_AT_STEP) / float(self.hyperparameters.LEARNING_RATE_DECAY_STEP)))
                next_learning_rate = current_learning_rate * learning_rate_decay
                self.tf_session.run(tf.assign(self.tf_graph_train_bundle.learning_rate, next_learning_rate))

            if training_step % self.hyperparameters.METRICS_SAVE_STEP == 0:
                evaluated_all_summaries = self.tf_session.run([self.tf_all_summaries], feed_dict=feed_dict)[0]
                self.tf_summary_writer.add_summary(evaluated_all_summaries, training_step)

            return evaluated_loss

    def _circular_selection_of_batch(self, offset, train_bundles, train_bundles_len):
        selection_end_of_list = train_bundles[offset:min(train_bundles_len, (offset + self.hyperparameters.SGD_BATCH_SIZE))]
        selection_beggining_of_list = train_bundles[0:max(0, ((offset + self.hyperparameters.SGD_BATCH_SIZE) - train_bundles_len))]
        return selection_end_of_list + selection_beggining_of_list

    def forward_pass_single(self, input_state):
        return self._forward_pass([input_state], self.tf_graph_forward_pass_bundle_single)

    def forward_pass_batched(self, input_states):
        return self._forward_pass(input_states, self.tf_graph_forward_pass_bundle_batched)

    def _forward_pass(self, input_states, forward_pass_graph_bundle):
        feed_dict = {forward_pass_graph_bundle.input_state: np.asarray(self._replace_non_existing_states_with_zeroed_states(input_states))}
        with self.tf_session.as_default():
            return self.tf_session.run(
                [forward_pass_graph_bundle.output_all_actions_q_values],
                feed_dict=feed_dict)[0]

    def _replace_non_existing_states_with_zeroed_states(self, states):
        result = [None] * len(states)
        for idx, state in enumerate(states):
            if state is None:
                result[idx] = np.zeros((self.screen_height, self.screen_width, self.num_channels))
            else:
                result[idx] = state
        return result

    def _build_assigns_train_to_forward_pass_variables(self):
        assigns = []
        with self.tf_graph.as_default():
            for variable in tf.all_variables():
                self._assign_forward_pass_variable_to_train_variable(forward_pass_prefix=self.tf_graph_forward_pass_bundle_single.variable_scope_name_prefix,
                                                                     variable=variable,
                                                                     assigns=assigns)
                self._assign_forward_pass_variable_to_train_variable(forward_pass_prefix=self.tf_graph_forward_pass_bundle_batched.variable_scope_name_prefix,
                                                                     variable=variable,
                                                                     assigns=assigns)
        return assigns

    def _assign_forward_pass_variable_to_train_variable(self, forward_pass_prefix, variable, assigns):
        if variable.name.startswith(forward_pass_prefix):
            forward_pass_variable = variable
            train_variable_name = forward_pass_variable.name.replace(forward_pass_prefix, self.tf_graph_train_bundle.variable_scope_name_prefix)
            train_variable = [v for v in tf.all_variables() if train_variable_name in v.name][0]
            assigns.append(forward_pass_variable.assign(train_variable))
            self.logger.debug("{target} will be assigned to {source} when summoned".format(target=forward_pass_variable.name,
                                                                               source=train_variable.name))

    def save(self, path):
        with self.tf_session.as_default():
            save_path = self.tf_saver.save(self.tf_session, path)
            self.logger.info("Q Network saved in file: %s" % save_path)

    def restore(self, path):
        with self.tf_session.as_default():
            self.tf_saver.restore(self.tf_session, path)
            self.logger.info("Q Network restored from file: %s" % path)

