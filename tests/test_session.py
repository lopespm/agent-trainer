from unittest import TestCase
from mock import Mock, MagicMock, patch

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING) # Supress excessive tesnorflow logging. Must be placed before the SessionRunner import

from agent.trainer.session import SessionRunner


class TestSessionRunner(TestCase):

    def test_add_metrics_bundle_on_first_episode(self):
        session = self._create_session()
        generic_hyperparameters = self._create_generic_parameters(NUM_EPISODES_TO_TRAIN=1)
        runner = self._create_session_runner(config=Mock(trained_using_aws_spot_instance=False),
                                             generic_hyperparameters=generic_hyperparameters)
        runner.train(session)

        self.assertEqual(session.metrics_trained_play.append.call_count, 1)

    def test_add_metrics_bundle_on_step(self):
        session = self._create_session()
        generic_hyperparameters =  self._create_generic_parameters(NUM_EPISODES_TO_TRAIN=4, EPISODE_TRAINED_PLAY_METRICS_GATHER_STEP=2)
        episode_runner = Mock()
        episode_runner.train.return_value = MagicMock()
        runner = self._create_session_runner(config=Mock(trained_using_aws_spot_instance=False),
                                             episode_runner=episode_runner,
                                             generic_hyperparameters=generic_hyperparameters)
        runner.train(session)

        self.assertEqual(session.metrics_trained_play.append.call_count, 2)

    def test_save_on_first_episode(self):
        session = self._create_session()
        generic_hyperparameters = self._create_generic_parameters(NUM_EPISODES_TO_TRAIN=1)
        runner = self._create_session_runner(config=Mock(trained_using_aws_spot_instance=False),
                                             generic_hyperparameters=generic_hyperparameters)
        runner.train(session)

        session.save.assert_called_once_with()

    @patch('agent.trainer.session.requests')
    def test_save_and_exits_when_scheduled_for_shutdown_on_aws_spot(self, mock_requests):
        mock_requests.get.return_value.status_code = 200
        session = self._create_session()
        generic_hyperparameters = self._create_generic_parameters(NUM_EPISODES_TO_TRAIN=10, EPISODE_SAVE_STEP=2)
        runner = self._create_session_runner(config=Mock(trained_using_aws_spot_instance=True),
                                             generic_hyperparameters=generic_hyperparameters)

        with self.assertRaises(SystemExit):
            runner.train(session)

        self.assertEqual(session.save.call_count, 1)

    @patch('agent.trainer.session.requests')
    def test_does_not_perform_request_when_not_aws_spot(self, mock_requests):
        session = self._create_session()
        generic_hyperparameters = self._create_generic_parameters(NUM_EPISODES_TO_TRAIN=1)
        runner = self._create_session_runner(config=Mock(trained_using_aws_spot_instance=False),
                                             generic_hyperparameters=generic_hyperparameters)

        runner.train(session)

        self.assertEqual(mock_requests.get.call_count, 0)


    def _create_session_runner(self, config, generic_hyperparameters, episode_runner=Mock()):
        runner = SessionRunner(config=config,
                               episode_runner=episode_runner,
                               generic_hyperparameters=generic_hyperparameters,
                               preprocessor_hyperparameters=Mock())
        return runner

    def _create_session(self, episode_number=0, replay_memories=MagicMock()):
        session = Mock()
        session.episode_number = episode_number
        session.replay_memories = replay_memories
        return session

    def _create_generic_parameters(self,
                                   NUM_EPISODES_TO_TRAIN,
                                   EPISODE_TRAINED_PLAY_METRICS_GATHER_STEP=1000,
                                   EPISODE_SAVE_STEP=1000):
        return Mock(NUM_EPISODES_TO_TRAIN=NUM_EPISODES_TO_TRAIN,
                    EPISODE_TRAINED_PLAY_METRICS_GATHER_STEP=EPISODE_TRAINED_PLAY_METRICS_GATHER_STEP,
                    EPISODE_SAVE_STEP=EPISODE_SAVE_STEP)