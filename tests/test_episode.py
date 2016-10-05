from unittest import TestCase

from mock import Mock

from agent.game.action import Action
from agent.trainer.episode import EpisodeActionlessBegginingPreventer


class TestEpisodeActionlessBegginingPreventer(TestCase):

    def test_prevent(self):
        no_action_preventer = EpisodeActionlessBegginingPreventer(Mock(MAXIMUM_NO_ACTIONS_BEGGINING_EPISODE=3))
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),                          Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),                          Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),                          Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction, random_action_seed=45),   Action.TurnLeft)

    def test_keep_preventing_until_different_action_is_issued(self):
        no_action_preventer = EpisodeActionlessBegginingPreventer(Mock(MAXIMUM_NO_ACTIONS_BEGGINING_EPISODE=3))
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),                          Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),                          Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),                          Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction, random_action_seed=45),   Action.TurnLeft)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction, random_action_seed=20),   Action.BrakeAndTurnRight)
        self.assertEqual(no_action_preventer.prevent(Action.Brake),                             Action.Brake)

    def test_do_not_prevent_if_not_in_beggining_sequence(self):
        no_action_preventer = EpisodeActionlessBegginingPreventer(Mock(MAXIMUM_NO_ACTIONS_BEGGINING_EPISODE=3))
        self.assertEqual(no_action_preventer.prevent(Action.Brake),     Action.Brake)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),  Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),  Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),  Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),  Action.NoAction)


    def test_do_not_prevent_if_beggining_sequence_has_action_before_limit_is_reached(self):
        no_action_preventer = EpisodeActionlessBegginingPreventer(Mock(MAXIMUM_NO_ACTIONS_BEGGINING_EPISODE=3))
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),      Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),      Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),      Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.Accelerate),    Action.Accelerate)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),      Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),      Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),      Action.NoAction)
        self.assertEqual(no_action_preventer.prevent(Action.NoAction),      Action.NoAction)
