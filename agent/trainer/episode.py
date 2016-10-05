import logging
import math
import random
import time
from collections import namedtuple
from itertools import count

import numpy as np

from agent.game.action import Action
from agent.game.cannonball_wrapper import CannonballFactory
from agent.hyperparameters import GenericHyperparameters, PreprocessorHyperparameters
from agent.trainer.image_preprocessor import ImagePreprocessor
from agent.trainer.replay_memories import ReplayMemory
from agent.trainer.visualization.metrics import MetricsInTrainAccumulator
from agent.trainer.visualization.metrics import MetricsTrainedPlayAccumulator
from agent.utils.utils import LinearInterpolator, NumberHistory

EpisodeTrainBundle = namedtuple("EpisodeTrainBundle", ["replay_memories", "q_network", "global_step", "metrics_bundle"])
EpisodePlayBundle = namedtuple("EpisodePlayBundle", ["states", "action_indexes", "metrics_bundle"])
GameSnapshot = namedtuple("GameSnapshot", ["score_history", "speed", "num_wheels_off_road", "crashed", "game_over",
                                           "delta_score", "log_speed", "log_delta_score", "reward"])

class EpisodeActionlessBegginingPreventer(object):
    def __init__(self, hyperparamenters):
        self.logger = logging.getLogger(__name__)
        self.hyperparamenters = hyperparamenters
        self.beggining_of_episode = True
        self.no_action_beggining_count = 0

    def prevent(self, action, random_action_seed=None):
        if not self.beggining_of_episode:
            return action
        elif action == Action.NoAction:
            return self._override_action_if_no_action_limit_reached(random_action_seed)
        else:
            self.beggining_of_episode = False
            return action

    def _override_action_if_no_action_limit_reached(self, random_action_seed):
        self.no_action_beggining_count = self.no_action_beggining_count + 1
        if self.no_action_beggining_count > self.hyperparamenters.MAXIMUM_NO_ACTIONS_BEGGINING_EPISODE:
            random.seed(random_action_seed)
            random_action = random.choice(self._actions_except_no_action())
            self.logger.info('Overriding action due to maximum NO_ACTIONs reached. New (exploratory) action: {action}'.format(
                action=random_action))
            return random_action
        else:
            return Action.NoAction

    def _actions_except_no_action(self):
        actions = list(Action)
        actions.remove(Action.NoAction)
        return actions

class RewardCalculator(object):
    MAX_DELTA_SCORE = 500000.0
    MAX_SPEED = 20000000.0

    def reward(self, log_speed, num_wheels_off_road, crashed):
        if num_wheels_off_road > 0 or crashed:
            return -0.6
        elif log_speed == 0.0:
            return -0.04
        else:
            return log_speed

    def delta_score(self, score_history):
        return score_history.current() - score_history.previous()

    def log_delta_score(self, delta_score):
        x = (float(delta_score) / float((self.MAX_DELTA_SCORE / 10.0)) * 100.0) + 0.99
        base = 10
        return math.log(x, base)

    def log_speed(self, speed):
        x = (float(speed) / float((self.MAX_SPEED)) * 100.0) + 0.99
        base = 10
        return max(0.0, math.log(x, base) / 2.0)


class EpisodeCommon(object):
    def __init__(self, game_wrapper, preprocessor, hyperparameters, reward_calc=RewardCalculator()):
        self.logger = logging.getLogger(__name__)
        self.game_wrapper = game_wrapper
        self.preprocessor = preprocessor
        self.hyperparameters = hyperparameters
        self.reward_calc = reward_calc

    def exploratory_action(self, random_action_seed=None):
        random.seed(random_action_seed)
        action = random.choice(list(Action))
        self.logger.info('Exploratory Action {action}'.format(action=action))
        return action

    def best_action_using_current_knowledge(self, actions_q_values):
        action_index = np.argmax(actions_q_values)
        action = Action(action_index)
        self.logger.info(actions_q_values)
        self.logger.info('Best Action {action}'.format(action=action))
        return action

    def current_game_frame(self):
        return self.preprocessor.process(self.game_wrapper.pixels_rgb())

    def initial_state_from_initial_frame(self):
        return np.concatenate([self.current_game_frame()] * self.hyperparameters.AGENT_HISTORY_LENGTH, axis=2)

    def skip_frames_and_get_consequent_state(self, action):
        agent_history = []
        for i in xrange(self.hyperparameters.FRAMES_SKIPPED_UNTIL_NEXT_ACTION):
            self.game_wrapper.tick(self.game_wrapper.create_action(action))
            if self.keep_frame_for_history(i):
                agent_history.append(self.current_game_frame())
        return np.concatenate(agent_history, axis=2)

    def keep_frame_for_history(self, i):
        return (self.hyperparameters.FRAMES_SKIPPED_UNTIL_NEXT_ACTION - 1 - i) % math.ceil(
            self.hyperparameters.FRAMES_SKIPPED_UNTIL_NEXT_ACTION / float(self.hyperparameters.AGENT_HISTORY_LENGTH)) == 0.0

    def create_game_snapshot(self, score_history):
        score_history = score_history
        speed = self.game_wrapper.speed()
        num_wheels_off_road = self.game_wrapper.num_wheels_off_road()
        crashed = self.game_wrapper.crashed()
        game_over = self.game_wrapper.game_over()

        delta_score = self.reward_calc.delta_score(score_history)
        log_speed = self.reward_calc.log_speed(speed)

        return GameSnapshot(score_history=score_history,
                            speed=speed,
                            num_wheels_off_road=num_wheels_off_road,
                            crashed=crashed,
                            game_over=game_over,
                            delta_score=delta_score,
                            log_speed = log_speed,
                            log_delta_score = self.reward_calc.log_delta_score(delta_score),
                            reward = self.reward_calc.reward(log_speed, num_wheels_off_road, crashed))


class EpisodeTrain(object):
    def __init__(self,
                 game_wrapper,
                 preprocessor,
                 metrics_accumulator,
                 hyperparameters=GenericHyperparameters(),
                 episode_common_class=EpisodeCommon,
                 epsilon_interpolator=LinearInterpolator()):
        self.logger = logging.getLogger(__name__)
        self.game_wrapper = game_wrapper
        self.preprocessor = preprocessor
        self.metrics_accumulator = metrics_accumulator
        self.hyperparameters = hyperparameters
        self.epsilon_interpolator = epsilon_interpolator
        self.common = episode_common_class(game_wrapper, preprocessor, hyperparameters)
        self.no_action_preventer = EpisodeActionlessBegginingPreventer(hyperparameters)

    def train(self, initial_replay_memories, initial_q_network, initial_global_step):
        replay_memories = initial_replay_memories
        q_network = initial_q_network

        initial_state = self.common.initial_state_from_initial_frame()
        score_history = NumberHistory()

        execution_start_time = time.time()

        for global_step in count(start=initial_global_step):
            self.logger.info("-----")
            self.logger.info("STEP: " + str(global_step))

            actions_q_values = q_network.forward_pass_single(initial_state)[0]
            action = self._next_action(actions_q_values, global_step)
            consequent_state = self.common.skip_frames_and_get_consequent_state(action)

            score_history.add(self.game_wrapper.score())
            game_snapshot = self.common.create_game_snapshot(score_history=score_history)
            self.logger.info(
                "log_delta_score: {0:<30} log_speed: {1:<30} reward: {2:>5}".format(game_snapshot.log_delta_score,
                                                                                    game_snapshot.log_speed,
                                                                                    game_snapshot.reward))

            replay_memories.append(ReplayMemory(initial_state=initial_state, action_index=action.value,
                                                consequent_reward=game_snapshot.reward,
                                                consequent_state=None if game_snapshot.game_over else consequent_state))

            loss = self._learn_from_memories(replay_memories, q_network, global_step)

            self.metrics_accumulator.add(reward=game_snapshot.reward, delta_score=game_snapshot.log_delta_score,
                                         speed=game_snapshot.log_speed, action_value=np.mean(actions_q_values),
                                         loss=loss)

            if game_snapshot.game_over:
                execution_time = time.time() - execution_start_time
                metrics_bundle = self.metrics_accumulator.bundle(final_score=score_history.current(),
                                                                 execution_time=execution_time)
                return EpisodeTrainBundle(replay_memories=replay_memories, q_network=q_network, global_step=global_step,
                                          metrics_bundle=metrics_bundle)
            else:
                initial_state = consequent_state

    def _next_action(self, actions_q_values, global_step):
        if self._exploring(global_step):
            return self.no_action_preventer.prevent(self.common.exploratory_action())
        else:
            return self.no_action_preventer.prevent(self.common.best_action_using_current_knowledge(actions_q_values))

    def _exploring(self, global_step):
        if self._pre_learning_stage(global_step):
            return True

        epsilon = self.epsilon_interpolator.interpolate_with_clip(x=global_step,
                                                                  x0=self.hyperparameters.REPLAY_MEMORIES_MINIMUM_SIZE_FOR_LEARNING,
                                                                  x1=self.hyperparameters.EXPLORATION_EPSILON_FULL_DEGRADATION_AT_STEP,
                                                                  y0=self.hyperparameters.EXPLORATION_INITIAL_EPSILON,
                                                                  y1=self.hyperparameters.EXPLORATION_FINAL_EPSILON)

        self.logger.info('Epsilon: {epsilon}'.format(epsilon=epsilon))
        return random.random() <= epsilon


    def _learn_from_memories(self, replay_memories, q_network, global_step):
        if self._pre_learning_stage(global_step):
            loss = 0.0
            return loss

        sampled_replay_memories = replay_memories.sample(sample_size=self.hyperparameters.REPLAY_MEMORIES_TRAIN_SAMPLE_SIZE,
                                                         recent_memories_span=self.hyperparameters.REPLAY_MEMORIES_RECENT_SAMPLE_SPAN)
        consequent_states = [replay_memory.consequent_state for replay_memory in sampled_replay_memories]
        max_q_consequent_states = np.nanmax(q_network.forward_pass_batched(consequent_states), axis=1)

        train_bundles = [None] * self.hyperparameters.REPLAY_MEMORIES_TRAIN_SAMPLE_SIZE
        discount_factor = self.hyperparameters.Q_UPDATE_DISCOUNT_FACTOR
        for idx, replay_memory in enumerate(sampled_replay_memories):
            target_action_q_value = float(self._q_target(replay_memory=replay_memory,
                                                         max_q_consequent_state=max_q_consequent_states[idx],
                                                         discount_factor=discount_factor))
            train_bundles[idx] = q_network.create_train_bundle(state=replay_memory.initial_state,
                                                               action_index=replay_memory.action_index,
                                                               target_action_q_value=target_action_q_value)

        loss = q_network.train(train_bundles, global_step - self.hyperparameters.REPLAY_MEMORIES_MINIMUM_SIZE_FOR_LEARNING)
        return loss

    def _q_target(self, replay_memory, max_q_consequent_state, discount_factor):
        if replay_memory.consequent_state is None:
            return replay_memory.consequent_reward
        else:
            return replay_memory.consequent_reward + (discount_factor * max_q_consequent_state)

    def _pre_learning_stage(self, global_step):
        return global_step < self.hyperparameters.REPLAY_MEMORIES_MINIMUM_SIZE_FOR_LEARNING


class EpisodePlay(object):

    def __init__(self,
                 game_wrapper,
                 preprocessor,
                 metrics_accumulator,
                 hyperparameters=GenericHyperparameters(),
                 episode_common_class=EpisodeCommon):
        self.logger = logging.getLogger(__name__)
        self.game_wrapper = game_wrapper
        self.preprocessor = preprocessor
        self.metrics_accumulator = metrics_accumulator
        self.hyperparameters = hyperparameters
        self.common = episode_common_class(game_wrapper, preprocessor, hyperparameters)

    def play(self, q_network):
        states = []
        action_indexes = []

        state = self.common.initial_state_from_initial_frame()
        score_history = NumberHistory()

        execution_start_time = time.time()

        while not self.game_wrapper.game_over():
            actions_q_values = q_network.forward_pass_single(state)[0]
            action = self.common.best_action_using_current_knowledge(actions_q_values)

            score_history.add(self.game_wrapper.score())
            game_snapshot = self.common.create_game_snapshot(score_history=score_history)

            states.append(state)
            action_indexes.append(action.value)

            self.metrics_accumulator.add(reward=game_snapshot.reward, delta_score=game_snapshot.log_delta_score,
                                         speed=game_snapshot.log_speed, action_value=np.max(actions_q_values))

            state = self.common.skip_frames_and_get_consequent_state(action)

        return EpisodePlayBundle(states=states,
                                 action_indexes=action_indexes,
                                 metrics_bundle=self.metrics_accumulator.bundle(final_score=score_history.current(),
                                                                                execution_time=time.time() - execution_start_time))


class EpisodeRunner(object):
    def __init__(self,
                 game_wrapper_factory=CannonballFactory(),
                 preprocessor=ImagePreprocessor(),
                 preprocessor_hyperparameters=PreprocessorHyperparameters()):
        self.game_wrapper_factory = game_wrapper_factory
        self.preprocessor = preprocessor
        self.preprocessor_hyperparameters = preprocessor_hyperparameters

    def play(self, q_network, game_config_filename="config_play.xml", episode_number=0):
        with self.game_wrapper_factory.create(game_config_filename) as game_wrapper:
            game_wrapper.reset()
            game_wrapper.start_game()
            episode = EpisodePlay(game_wrapper=game_wrapper,
                                  preprocessor=self.preprocessor,
                                  metrics_accumulator=MetricsTrainedPlayAccumulator(episode_number))
            return episode.play(q_network)


    def train(self, episode_number, session, game_config_filename="config_train.xml"):
        with self.game_wrapper_factory.create(game_config_filename) as game_wrapper:
            game_wrapper.start_game()
            episode = EpisodeTrain(game_wrapper=game_wrapper,
                                   preprocessor=self.preprocessor,
                                   metrics_accumulator=MetricsInTrainAccumulator(episode_number))
            return episode.train(initial_replay_memories=session.replay_memories,
                                 initial_q_network=session.q_network,
                                 initial_global_step=session.global_step)