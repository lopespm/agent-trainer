import os
import unittest

from tempdir import TempDir

from agent.trainer.visualization.metrics import Metrics, MetricsInTrainAccumulator, \
    MetricsInTrainBundle, MetricsTrainedPlayAccumulator, MetricsTrainedPlayBundle


class TestMetricsInTrain(unittest.TestCase):

    def test_bundle(self):
        metrics = MetricsInTrainAccumulator(episode_number=31)
        metrics.add(reward=59.2, delta_score=33.0, speed=20.0, action_value=49.0, loss=63.0)
        bundle = metrics.bundle(final_score=888.0, execution_time=220.0)

        expected_bundle = MetricsInTrainBundle(episode_number=31, average_reward=59.2, average_delta_score=33.0,
                                               average_speed=20.0, average_action_value=49.0, average_loss=63.0,
                                               final_score=888.0, execution_time=220.0)

        self.assertEqual(bundle, expected_bundle)

    def test_bundle_averages(self):
        metrics = MetricsInTrainAccumulator(episode_number=31)
        metrics.add(reward=59.2, delta_score=33.0, speed=20.0, action_value=49.0, loss=63.0)
        metrics.add(reward=38.4, delta_score=67.0, speed=94.0, action_value=0.1, loss=2087773.0)

        bundle = metrics.bundle(final_score=888.0, execution_time=114.0)

        expected_bundle = MetricsInTrainBundle(episode_number=31, average_reward=48.8, average_delta_score=50.0,
                                               average_speed=57.0, average_action_value=24.55, average_loss=1043918.0,
                                               final_score=888.0, execution_time=114.0)

        self.assertEqual(bundle, expected_bundle)

    def test_bundle_averages_with_undefined_values(self):
        metrics = MetricsInTrainAccumulator(episode_number=31)
        metrics.add(reward=59.2, delta_score=33.0, speed=20.0, action_value=49.0, loss=None)
        metrics.add(reward=38.4, delta_score=67.0, speed=94.0, action_value=3.5, loss=2087773.0)

        bundle = metrics.bundle(final_score=888.0, execution_time=114.0)

        expected_bundle = MetricsInTrainBundle(episode_number=31, average_reward=48.8, average_delta_score=50.0,
                                               average_speed=57.0, average_action_value=26.25, average_loss=2087773.0,
                                               final_score=888.0, execution_time=114.0)

        self.assertEqual(bundle, expected_bundle)

    def test_always_output_averages_as_float(self):
        metrics = MetricsInTrainAccumulator(episode_number=31)
        metrics.add(reward=44, delta_score=33, speed=20, action_value=49, loss=63)
        bundle = metrics.bundle(final_score=888.0, execution_time=114.0)

        self.assertTrue(isinstance(bundle.average_reward, float))
        self.assertTrue(isinstance(bundle.average_delta_score, float))
        self.assertTrue(isinstance(bundle.average_speed, float))
        self.assertTrue(isinstance(bundle.average_action_value, float))
        self.assertTrue(isinstance(bundle.average_loss, float))

class TestMetricsTrainedPlay(unittest.TestCase):

    def test_bundle(self):
        metrics = MetricsTrainedPlayAccumulator(episode_number=31)
        metrics.add(reward=46.8, delta_score=33.0, speed=20.0, action_value=49.0)
        bundle = metrics.bundle(final_score=888.0, execution_time= 289.0)

        expected_bundle = MetricsTrainedPlayBundle(episode_number=31, average_reward=46.8, average_delta_score=33.0,
                                                   average_speed=20.0, average_action_value=49.0, final_score=888.0,
                                                   execution_time= 289.0)

        self.assertEqual(bundle, expected_bundle)

    def test_bundle_averages(self):
        metrics = MetricsTrainedPlayAccumulator(episode_number=31)
        metrics.add(reward=59.2, delta_score=33.0, speed=20.0, action_value=49.0)
        metrics.add(reward=38.4, delta_score=67.0, speed=94.0, action_value=0.1)

        bundle = metrics.bundle(final_score=888.0, execution_time= 289.0)

        expected_bundle = MetricsTrainedPlayBundle(episode_number=31, average_reward=48.8, average_delta_score=50.0,
                                                   average_speed=57.0, average_action_value=24.55, final_score=888.0,
                                                   execution_time= 289.0)

        self.assertEqual(bundle, expected_bundle)

    def test_bundle_averages_with_undefined_values(self):
        metrics = MetricsTrainedPlayAccumulator(episode_number=31)
        metrics.add(reward=59.2, delta_score=33.0, speed=None, action_value=49.0)
        metrics.add(reward=38.4, delta_score=67.0, speed=94.0, action_value=3.5)

        bundle = metrics.bundle(final_score=888.0, execution_time= 289.0)

        expected_bundle = MetricsTrainedPlayBundle(episode_number=31, average_reward=48.8, average_delta_score=50.0,
                                                   average_speed=94.0, average_action_value=26.25, final_score=888.0,
                                                   execution_time= 289.0)

        self.assertEqual(bundle, expected_bundle)

    def test_always_output_averages_as_float(self):
        metrics = MetricsTrainedPlayAccumulator(episode_number=31)
        metrics.add(reward=49, delta_score=33, speed=20, action_value=49)
        bundle = metrics.bundle(final_score=888.0, execution_time= 289.0)

        self.assertTrue(isinstance(bundle.average_reward, float))
        self.assertTrue(isinstance(bundle.average_delta_score, float))
        self.assertTrue(isinstance(bundle.average_speed, float))
        self.assertTrue(isinstance(bundle.average_action_value, float))

class TestMetrics(unittest.TestCase):

    def test_persist_and_retrieve(self):
        bundle1 = self._create_metrics_bundle(episode_number=31, average_delta_score=33.0,
                                              average_speed=20.0, average_action_value=49.0,
                                              average_loss=63.0, final_score=888.0, execution_time=114.0)
        bundle2 = self._create_metrics_bundle(episode_number=32, average_delta_score=123.0,
                                              average_speed=3.55, average_action_value=312.1,
                                              average_loss=11.0, final_score=1002.0, execution_time=114.0)

        with TempDir() as temp_directory:
            metrics1_file_name = os.path.join(temp_directory, 'metrics.dat')
            metrics1 = Metrics(metrics_path=metrics1_file_name, bundler=MetricsInTrainBundle)
            metrics1.append(bundle1)
            metrics1.append(bundle2)
            metrics1.persist_and_flush_memory()

            metrics2 = Metrics(metrics_path=metrics1_file_name, bundler=MetricsInTrainBundle)
            all_episode_metrics = metrics2.all_metric_bundles()

            self.assertSequenceEqual(all_episode_metrics, [bundle1, bundle2])

    def test_persist_and_retrieve_undefined_value(self):
        bundle1 = self._create_metrics_bundle(average_speed=None)

        with TempDir() as temp_directory:
            metrics1_file_name = os.path.join(temp_directory, 'metrics.dat')
            metrics1 = Metrics(metrics_path=metrics1_file_name, bundler=MetricsInTrainBundle)
            metrics1.append(bundle1)
            metrics1.persist_and_flush_memory()

            metrics2 = Metrics(metrics_path=metrics1_file_name, bundler=MetricsInTrainBundle)
            all_episode_metrics = metrics2.all_metric_bundles()

            self.assertSequenceEqual(all_episode_metrics, [bundle1])

    def test_persist_and_flush_in_memory_metrics(self):
        bundle1 = self._create_metrics_bundle(average_action_value=321.0)
        bundle2 = self._create_metrics_bundle(average_action_value=421.0)

        with TempDir() as temp_directory:
            metrics_file_name = os.path.join(temp_directory, 'metrics.dat')
            metrics = Metrics(metrics_path=metrics_file_name, bundler=MetricsInTrainBundle)
            metrics.append(bundle1)
            metrics.append(bundle2)
            metrics.persist_and_flush_memory()
            metrics.persist_and_flush_memory()

            episode_metrics = metrics.all_metric_bundles()

            self.assertSequenceEqual(episode_metrics, [bundle1, bundle2])

    def _create_metrics_bundle(self, episode_number=31, average_reward=33.4,
                               average_delta_score=33.0, average_speed=20.0,
                               average_action_value=49.0, average_loss=63.0,
                               final_score=888.0, execution_time=114.0):
        return MetricsInTrainBundle(episode_number=episode_number, average_reward=average_reward,
                                    average_delta_score=average_delta_score, average_speed=average_speed,
                                    average_action_value=average_action_value, average_loss=average_loss,
                                    final_score=final_score, execution_time=execution_time)


