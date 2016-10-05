from collections import namedtuple
import matplotlib.pyplot as plt
import jsonpickle
import numpy as np

from agent.trainer.visualization import style

ValuesGroup = namedtuple("ValuesGroup", ["values", "label"])
class MetricsPlot(object):

    def __init__(self):
        self.tableau20 = style.generate_tableau20_colors()

    def show(self, x_group, y_groups):
        self._plot(x_group, y_groups)
        plt.show()

    def save_to_image(self, x_group, y_groups, image_path):
        self._plot(x_group, y_groups)
        plt.savefig(image_path)

    def _plot(self, x_group, y_groups):
        figure, axarr = plt.subplots(len(y_groups), sharex=True, figsize=(10, 12))
        figure.subplots_adjust(hspace=.0)

        for index, y_group in enumerate(y_groups):
            axarr[index].plot(x_group.values, y_group.values, color=self.tableau20[index])
            axarr[index].set_ylabel(y_group.label)
            axarr[index].get_yaxis().set_label_coords(-0.085, 0.5)

        axarr[-1].set_xlabel(x_group.label)
        axarr[-1].get_xaxis().set_label_coords(0.5, -0.23)


class MetricsBundle(object):
    def __init__(self,
                 episode_number,
                 average_reward,
                 average_delta_score,
                 average_speed,
                 average_action_value,
                 final_score,
                 execution_time):
        self.episode_number = int(episode_number)
        self.average_reward = self._float_or_none(average_reward)
        self.average_delta_score = self._float_or_none(average_delta_score)
        self.average_speed = self._float_or_none(average_speed)
        self.average_action_value = self._float_or_none(average_action_value)
        self.final_score = float(final_score)
        self.execution_time = float(execution_time)

    def _float_or_none(self, value):
        if (value is None):
            return None
        else:
            return float(value)

    @staticmethod
    def to_json(bundle):
        return jsonpickle.encode(bundle)

    @staticmethod
    def from_json(bundle_json):
        return jsonpickle.decode(bundle_json)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

class MetricsInTrainBundle(MetricsBundle):
    def __init__(self,
                 episode_number,
                 average_reward,
                 average_delta_score,
                 average_speed,
                 average_action_value,
                 average_loss,
                 final_score,
                 execution_time):
        super(MetricsInTrainBundle, self).__init__(episode_number=episode_number,
                                                   average_reward=average_reward,
                                                   average_delta_score=average_delta_score,
                                                   average_speed=average_speed,
                                                   average_action_value=average_action_value,
                                                   final_score=final_score,
                                                   execution_time=execution_time)
        self.average_loss = self._float_or_none(average_loss)

    @staticmethod
    def plot_value_groups(bundles):
        initial_x = 1
        x_group = ValuesGroup(label='Episodes', values=[bundle.episode_number for bundle in bundles][initial_x:])
        y_groups = [
            ValuesGroup(label='Average Reward', values=[bundle.average_reward for bundle in bundles][initial_x:]),
            ValuesGroup(label='Average Delta Score', values=[bundle.average_delta_score for bundle in bundles][initial_x:]),
            ValuesGroup(label='Average Speed', values=[bundle.average_speed for bundle in bundles][initial_x:]),
            ValuesGroup(label='Average Action Value', values=[bundle.average_action_value for bundle in bundles][initial_x:]),
            ValuesGroup(label='Average Loss', values=[bundle.average_loss for bundle in bundles][initial_x:]),
            ValuesGroup(label='Final Score', values=[bundle.final_score for bundle in bundles][initial_x:]),
            ValuesGroup(label='Execution Time (seconds)', values=[bundle.execution_time for bundle in bundles][initial_x:])]
        return x_group, y_groups


class MetricsTrainedPlayBundle(MetricsBundle):
    def __init__(self,
                 episode_number,
                 average_reward,
                 average_delta_score,
                 average_speed,
                 average_action_value,
                 final_score,
                 execution_time):
        super(MetricsTrainedPlayBundle, self).__init__(episode_number=episode_number,
                                                       average_reward=average_reward,
                                                       average_delta_score=average_delta_score,
                                                       average_speed=average_speed,
                                                       average_action_value=average_action_value,
                                                       final_score=final_score,
                                                       execution_time=execution_time)

    @staticmethod
    def plot_value_groups(bundles):
        initial_x = 1
        x_group = ValuesGroup(label='Episodes', values=[bundle.episode_number for bundle in bundles][initial_x:])
        y_groups = [
            ValuesGroup(label='Average Reward', values=[bundle.average_reward for bundle in bundles][initial_x:]),
            ValuesGroup(label='Average Delta Score', values=[bundle.average_delta_score for bundle in bundles][initial_x:]),
            ValuesGroup(label='Average Speed', values=[bundle.average_speed for bundle in bundles][initial_x:]),
            ValuesGroup(label='Average Action Value', values=[bundle.average_action_value for bundle in bundles][initial_x:]),
            ValuesGroup(label='Final Score', values=[bundle.final_score for bundle in bundles][initial_x:])]
        return x_group, y_groups


class MetricsAccumulator(object):

    def __init__(self, episode_number):
        self._episode_number = episode_number
        self._rewards = []
        self._delta_scores = []
        self._speeds = []
        self._action_values = []

    def _average(self, values):
        values_without_nones = [float(value) for value in values if value is not None]
        return np.mean(values_without_nones)


class MetricsInTrainAccumulator(MetricsAccumulator):

    def __init__(self, episode_number):
        super(MetricsInTrainAccumulator, self).__init__(episode_number)
        self._losses = []

    def add(self, reward, delta_score, speed, action_value, loss):
        self._rewards.append(reward)
        self._delta_scores.append(delta_score)
        self._speeds.append(speed)
        self._action_values.append(action_value)
        self._losses.append(loss)

    def bundle(self, final_score, execution_time):
        return MetricsInTrainBundle(episode_number=self._episode_number,
                                    average_reward=self._average(self._rewards),
                                    average_delta_score=self._average(self._delta_scores),
                                    average_speed=self._average(self._speeds),
                                    average_action_value=self._average(self._action_values),
                                    average_loss=self._average(self._losses),
                                    final_score=final_score,
                                    execution_time=execution_time)


class MetricsTrainedPlayAccumulator(MetricsAccumulator):

    def __init__(self, episode_number):
        super(MetricsTrainedPlayAccumulator, self).__init__(episode_number)

    def add(self, reward, delta_score, speed, action_value):
        self._rewards.append(reward)
        self._delta_scores.append(delta_score)
        self._speeds.append(speed)
        self._action_values.append(action_value)

    def bundle(self, final_score, execution_time):
        return MetricsTrainedPlayBundle(episode_number=self._episode_number,
                                        average_reward=self._average(self._rewards),
                                        average_delta_score=self._average(self._delta_scores),
                                        average_speed=self._average(self._speeds),
                                        average_action_value=self._average(self._action_values),
                                        final_score=final_score,
                                        execution_time = execution_time)


class Metrics(object):

    def __init__(self, metrics_path, bundler, metrics_plot=MetricsPlot()):
        self.metrics_path = metrics_path
        self.bundler = bundler
        self.metrics_plot = metrics_plot
        self.partial_metrics_episode_bundles = []

    def append(self, metrics_episode_bundle):
        self.partial_metrics_episode_bundles.append(metrics_episode_bundle)

    def persist_and_flush_memory(self):
        with open(self.metrics_path, 'a') as file:
            for bundle in self.partial_metrics_episode_bundles:
                bundle_json = self.bundler.to_json(bundle)
                file.write(bundle_json + '\n')
        del self.partial_metrics_episode_bundles[:]

    def all_metric_bundles(self):
        return self._load_metrics()

    def save_plot_to_image(self, image_path):
        x_group, y_groups = self.bundler.plot_value_groups(self._load_metrics())
        self.metrics_plot.save_to_image(x_group, y_groups, image_path)

    def show_plot(self):
        x_group, y_groups = self.bundler.plot_value_groups(self._load_metrics())
        self.metrics_plot.show(x_group, y_groups)

    def _load_metrics(self):
        all_metrics_episode_bundles = []
        with open(self.metrics_path, 'r') as file:
            for line in file:
                if line and line != '\n':
                    all_metrics_episode_bundles.append(self.bundler.from_json(line))

        return all_metrics_episode_bundles