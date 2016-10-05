import cPickle as pickle
import logging
import os
from datetime import datetime as dt

import requests

from agent.game.action import Action
from agent.hyperparameters import GenericHyperparameters, PreprocessorHyperparameters
from agent.trainer.episode import EpisodeRunner
from agent.trainer.q_network import QNetworkFactory
from agent.trainer.replay_memories import ReplayMemories
from agent.trainer.visualization.tsne import Tsne
from agent.trainer.visualization.metrics import Metrics, MetricsInTrainBundle, MetricsTrainedPlayBundle


class SessionRunner(object):

    def __init__(self,
                 config,
                 episode_runner=EpisodeRunner(),
                 generic_hyperparameters=GenericHyperparameters(),
                 preprocessor_hyperparameters=PreprocessorHyperparameters()):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.episode_runner = episode_runner
        self.generic_hyperparameters = generic_hyperparameters
        self.preprocessor_hyperparameters = preprocessor_hyperparameters

    def play(self, session_id):
        session = Session(config=self.config, session_id=session_id)
        session.restore_q_network()
        self.episode_runner.play(session.q_network)

    def play_and_visualize_q_network_tsne(self, session_id):
        tsne = Tsne(number_of_images_per_input=self.generic_hyperparameters.AGENT_HISTORY_LENGTH,
                    single_input_image_width_input=self.preprocessor_hyperparameters.OUTPUT_WIDTH,
                    output_descriptor_enum=Action)
        session = Session(config=self.config, session_id=session_id)
        tsne.init()
        self.logger.info("Playing through an episode to gather input_states and resulting action_indexes")
        session.restore_q_network()
        play_bundle = self.episode_runner.play(session.q_network)
        tsne.save_visualization_to_image(inputs=play_bundle.states,
                                         outputs=play_bundle.action_indexes,
                                         folder_path_for_result_image=session.visualizations_folder_path())

    def train_new(self):
        self.train(Session(config=self.config))

    def train_resume(self, session_id):
        session = Session(config=self.config, session_id=session_id)
        session.restore()
        self.train(session)

    def train(self, session):
        self.logger.info("Session id {0}: training".format(session.session_id))
        for episode_number in xrange(session.episode_number, self.generic_hyperparameters.NUM_EPISODES_TO_TRAIN):

            self.logger.info("Session {0} at episode {1}: q network ref{2}; replay memories len {3}; final iteration: {4}"
                             .format(session.session_id, episode_number, session.q_network, len(session.replay_memories), session.global_step))

            train_bundle = self.episode_runner.train(episode_number, session)
            session.q_network = train_bundle.q_network
            session.replay_memories = train_bundle.replay_memories
            session.global_step = train_bundle.global_step
            session.episode_number = episode_number + 1
            session.metrics_in_train.append(train_bundle.metrics_bundle)

            if (episode_number % self.generic_hyperparameters.EPISODE_TRAINED_PLAY_METRICS_GATHER_STEP == 0):
                play_bundle = self.episode_runner.play(session.q_network, episode_number=episode_number)
                session.metrics_trained_play.append(play_bundle.metrics_bundle)

            if self._host_scheduled_for_shutdown():
                session.save()
                raise SystemExit("Host is scheduled for shutdown. Saved state and terminated training.")
            elif (episode_number % self.generic_hyperparameters.EPISODE_SAVE_STEP == 0):
                session.save()

    def _host_scheduled_for_shutdown(self):
        if self.config.trained_using_aws_spot_instance:
            # More information here: https://aws.amazon.com/blogs/aws/new-ec2-spot-instance-termination-notices/
            result = requests.get("http://169.254.169.254/latest/meta-data/spot/termination-time", timeout=1)
            return result.status_code != 404
        else:
            return False


class SessionMetricsPresenter(object):
    def __init__(self, config):
        self.config = config

    def show(self, session_id):
        session = Session(config=self.config, session_id=session_id)
        session.show_metrics()

    def save_to_image(self, session_id):
        session = Session(config=self.config, session_id=session_id)
        session.save_metrics_to_image()


def _generate_session_id():
    return str(dt.now().strftime("%Y%m%d%H%M"))

class Session(object):

    def __init__(self, config, session_id=_generate_session_id()):
        self.logger = logging.getLogger(__name__)
        self._repository = SessionRepository(session_id=session_id, train_results_path=config.train_results_root_folder)
        self.session_id = session_id
        self.q_network = self._repository.q_network()
        self.replay_memories = self._repository.new_replay_memories()
        self.global_step = 0
        self.episode_number = 0
        self.metrics_in_train = self._repository.new_metrics_in_train()
        self.metrics_trained_play = self._repository.new_metrics_trained_play()

    def save(self):
        self.logger.info("Starting to save session {0}".format(self.session_id))
        self._repository.save_replay_memories(self.replay_memories)
        self._repository.save_q_network(self.q_network)
        self._repository.save_global_step(self.global_step)
        self._repository.save_episode_number(self.episode_number)
        self._repository.save_metrics(self.metrics_in_train)
        self._repository.save_metrics(self.metrics_trained_play)
        self.logger.info("Finished saving session {0}".format(self.session_id))

    def restore(self):
        self.logger.info("Restoring session {0}".format(self.session_id))
        self.q_network = self._repository.load_q_network()
        self.replay_memories = self._repository.load_replay_memories()
        self.global_step = self._repository.load_global_step()
        self.episode_number = self._repository.load_episode_number()

    def restore_q_network(self):
        self.q_network = self._repository.load_q_network()

    def show_metrics(self):
        self.metrics_in_train.show_plot()
        self.metrics_trained_play.show_plot()

    def save_metrics_to_image(self):
        self.metrics_in_train.save_plot_to_image(self._repository.metrics_in_train_image_path())
        self.metrics_trained_play.save_plot_to_image(self._repository.metrics_trained_play_image_path())

    def visualizations_folder_path(self):
        return self._repository.visualizations_folder_path()


class SessionRepository(object):

    ASSET_REPLAY_MEMORIES = "replay_memories"
    ASSET_Q_NETWORK = "q_network.data"
    ASSET_GLOBAL_STEP = "global_step.pickle"
    ASSET_EPISODE_NUMBER = "episode_number.pickle"
    ASSET_METRICS_IN_TRAIN = "metrics_in_train.data"
    ASSET_METRICS_IN_TRAIN_IMAGE = "metrics_in_train.png"
    ASSET_METRICS_TRAINED_PLAY = "metrics_trained_play.data"
    ASSET_METRICS_TRAINED_PLAY_IMAGE = "metrics_trained_play.png"

    FOLDER_NAME_REPLAY_MEMORIES = "replay-memories"
    FOLDER_NAME_METRICS_SESSION = "metrics-session"
    FOLDER_NAME_METRICS_Q_NETWORK = "metrics-q-network"
    FOLDER_NAME_VISUALIZATIONS = "visualizations"

    def __init__(self,
                 session_id,
                 train_results_path,
                 generic_hyperparameters=GenericHyperparameters(),
                 preprocessor_hyperparameters=PreprocessorHyperparameters(),
                 q_network_factory=QNetworkFactory()):
        self.session_id = session_id
        self.generic_hyperparameters = generic_hyperparameters
        self.preprocessor_hyperparameters = preprocessor_hyperparameters
        self._q_network_factory = q_network_factory
        self._root_folder = train_results_path

    def q_network(self):
        return self._q_network_factory.create(screen_width=self.preprocessor_hyperparameters.OUTPUT_HEIGHT,
                                              screen_height=self.preprocessor_hyperparameters.OUTPUT_HEIGHT,
                                              num_channels=self.preprocessor_hyperparameters.OUTPUT_NUM_CHANNELS * self.generic_hyperparameters.AGENT_HISTORY_LENGTH,
                                              num_actions=len(Action),
                                              metrics_directory=self._session_folder() + "/" + self.FOLDER_NAME_METRICS_Q_NETWORK,
                                              batched_forward_pass_size=self.generic_hyperparameters.REPLAY_MEMORIES_TRAIN_SAMPLE_SIZE)
    def save_q_network(self, q_network):
        self._create_session_folder()
        q_network.save(self._path_for(self.ASSET_Q_NETWORK))

    def load_q_network(self):
        q_network = self.q_network()
        q_network.restore(self._path_for(self.ASSET_Q_NETWORK))
        return q_network


    def new_replay_memories(self):
        return ReplayMemories(previous_memories_path=self.replay_memories_path(),
                              max_current_memories_in_ram=self.generic_hyperparameters.MAX_REPLAY_MEMORIES_IN_RAM,
                              max_previous_memories_per_file=self.generic_hyperparameters.MAX_REPLAY_MEMORIES_PER_FILE)

    def save_replay_memories(self, replay_memories):
        self._create_session_folder()
        replay_memories.save(purge_min_recent_memories_to_keep=self.generic_hyperparameters.REPLAY_MEMORIES_RECENT_SAMPLE_SPAN)

    def load_replay_memories(self):
        return ReplayMemories(previous_memories_path=self.replay_memories_path(),
                              max_current_memories_in_ram=self.generic_hyperparameters.MAX_REPLAY_MEMORIES_IN_RAM,
                              max_previous_memories_per_file=self.generic_hyperparameters.MAX_REPLAY_MEMORIES_PER_FILE,
                              prefetch_into_ram=True)

    def replay_memories_path(self):
        folder = os.path.join(self._session_folder(), self.FOLDER_NAME_REPLAY_MEMORIES)
        if not os.path.exists(folder):
            os.makedirs(folder)
        return os.path.join(folder, self.ASSET_REPLAY_MEMORIES)


    def save_global_step(self, global_step):
        self._create_session_folder()
        self._save_asset(self.ASSET_GLOBAL_STEP, global_step)

    def load_global_step(self):
        return self._load_asset(self.ASSET_GLOBAL_STEP)


    def save_episode_number(self, episode_number):
        self._create_session_folder()
        self._save_asset(self.ASSET_EPISODE_NUMBER, episode_number)

    def load_episode_number(self):
        return self._load_asset(self.ASSET_EPISODE_NUMBER)


    def new_metrics_in_train(self):
        return self._new_metrics(self.ASSET_METRICS_IN_TRAIN, MetricsInTrainBundle)

    def metrics_in_train_image_path(self):
        return self._metrics_image_path(self.ASSET_METRICS_IN_TRAIN_IMAGE)

    def new_metrics_trained_play(self):
        return self._new_metrics(self.ASSET_METRICS_TRAINED_PLAY, MetricsTrainedPlayBundle)

    def metrics_trained_play_image_path(self):
        return self._metrics_image_path(self.ASSET_METRICS_TRAINED_PLAY_IMAGE)

    def _new_metrics(self, asset_filename, bundler):
        return Metrics(metrics_path=self._path_for(asset_filename), bundler=bundler)

    def _metrics_image_path(self, asset_filename):
        folder_path = os.path.join(self._session_folder(), self.FOLDER_NAME_METRICS_SESSION)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return os.path.join(folder_path, asset_filename)

    def save_metrics(self, metrics):
        metrics.persist_and_flush_memory()


    def visualizations_folder_path(self):
        folder_path = os.path.join(self._session_folder(), self.FOLDER_NAME_VISUALIZATIONS)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path


    def _session_folder(self):
        return os.path.join(self._root_folder, self.session_id)

    def _path_for(self, asset_filename):
        return os.path.join(self._session_folder(), asset_filename)

    def _create_session_folder(self):
        session_folder = self._session_folder()
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)

    def _save_asset(self, asset_filename, asset_object):
        with open(self._path_for(asset_filename), 'wb') as handle:
            pickle.dump(asset_object, handle, protocol=2)

    def _load_asset(self, asset_filename):
        with open(self._path_for(asset_filename), 'rb') as handle:
            return pickle.load(handle)
