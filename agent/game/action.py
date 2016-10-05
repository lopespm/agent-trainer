from enum import Enum

class Action(Enum):
    NoAction, \
    Accelerate, \
    Brake, \
    TurnLeft, \
    TurnRight, \
    AccelerateAndTurnLeft, \
    AccelerateAndTurnRight, \
    BrakeAndTurnLeft, \
    BrakeAndTurnRight = range(9)