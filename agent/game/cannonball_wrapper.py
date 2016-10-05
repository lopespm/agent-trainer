from ctypes import *
from shutil import copyfile

from agent.game.action import Action

lib = cdll.LoadLibrary('./lib/libcannonball.so')

import numpy as np


class CannonballFactory(object):
    def create(self, config_file_name): return Cannonball(config_file_name)

class Cannonball(object):

    FRAMES_SKIPPED_UNTIL_GAME_START = 145

    def __init__(self, config_file_name):
        copyfile(config_file_name, "config.xml")

    def __enter__(self):
        self.obj = lib.ExternalInterface_new()
        lib.ExternalInterface_init(self.obj)
        self.screen_width = lib.ExternalInterface_getScreenWidth(self.obj)
        self.screen_height = lib.ExternalInterface_getScreenHeight(self.obj)
        self.pixels_length = self.screen_width * self.screen_height
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self.obj = None
        return False

    def reset(self):
        lib.ExternalInterface_reset(self.obj)

    def close (self):
        lib.ExternalInterface_close(self.obj)

    def tick(self, cannonball_action):
        lib.ExternalInterface_tick(self.obj, cannonball_action)

    def pixels_rgb(self):
        f = lib.ExternalInterface_getPixelsRGB
        f.restype = POINTER(c_uint32)
        pixels_pointer = f(self.obj)
        # Each 32 bit (4 byte) integer has the following structure:
        # 8 bit unused, 8 bit Red, 8 bit Green, 8 bit Blue. In hex structure: 00 RR GG BB
        rgb_integers = np.ctypeslib.as_array(pixels_pointer, shape=(self.pixels_length, 1)).copy()
        rgb_components = self._rgb_integers_to_components(rgb_integers)
        return rgb_components.reshape((self.screen_height, self.screen_width, 3))

    # Transforms an array of integers, each representing the full RGB representation (x rows, 1 column)
    # to an array of separate red, green and blue components (x rows, 3 columns)
    def _rgb_integers_to_components(self, rgb_integers):
        red_mask = 0x00FF0000
        green_mask = 0x0000FF00
        blue_mask =  0x000000FF
        masks = np.asarray([[red_mask, green_mask, blue_mask]])
        masked_rgb_components = np.bitwise_and(rgb_integers, masks)

        red_shifted = np.right_shift(masked_rgb_components[:,0], 16)
        green_shifted = np.right_shift(masked_rgb_components[:,1], 8)
        blue_shifted =  np.right_shift(masked_rgb_components[:,2], 0)
        return np.array([red_shifted, green_shifted, blue_shifted]).transpose()

    def score(self):
        f = lib.ExternalInterface_getScore
        f.restype = c_uint32
        return float(max(0, f(self.obj)))

    def speed(self):
        f = lib.ExternalInterface_getSpeed
        f.restype = c_uint32
        return float(max(0, f(self.obj)))

    def num_wheels_off_road(self):
        f = lib.ExternalInterface_numWheelsOffRoad
        f.restype = c_uint32
        return f(self.obj)

    def crashed(self):
        f = lib.ExternalInterface_isCrashed
        f.restype = c_bool
        return f(self.obj)

    def game_over(self):
        f = lib.ExternalInterface_isGameOver
        f.restype = c_bool
        return f(self.obj)

    def start_game(self):
        action_start = CannonballAction()
        action_start.is_start_pressed = True
        for i in range(self.FRAMES_SKIPPED_UNTIL_GAME_START):
            self.tick(action_start)

    def create_action(self, action):
        cannonball_action = CannonballAction()
        if action == Action.Accelerate:
            cannonball_action.is_accel_pressed = True
        elif action == Action.Brake:
            cannonball_action.is_brake_pressed = True
        elif action == Action.TurnLeft:
            cannonball_action.is_steer_left_pressed = True
        elif action == Action.TurnRight:
            cannonball_action.is_steer_right_pressed = True
        elif action == Action.AccelerateAndTurnLeft:
            cannonball_action.is_accel_pressed = True
            cannonball_action.is_steer_left_pressed = True
        elif action == Action.AccelerateAndTurnRight:
            cannonball_action.is_accel_pressed = True
            cannonball_action.is_steer_right_pressed = True
        elif action == Action.BrakeAndTurnLeft:
            cannonball_action.is_brake_pressed = True
            cannonball_action.is_steer_left_pressed = True
        elif action == Action.BrakeAndTurnRight:
            cannonball_action.is_brake_pressed = True
            cannonball_action.is_steer_right_pressed = True
        return cannonball_action


class CannonballAction(Structure):
    _fields_ = [
        ("is_accel_pressed", c_bool),
        ("is_brake_pressed", c_bool),
        ("is_steer_left_pressed", c_bool),
        ("is_steer_right_pressed", c_bool),
        ("is_coin_pressed", c_bool),
        ("is_start_pressed", c_bool)]