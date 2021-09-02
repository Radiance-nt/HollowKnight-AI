# Define the actions we may need during training
# You can define your actions here

from Tool.SendKey import PressKey, ReleaseKey
import time

# Hash code for key we may use: https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes?redirectedfrom=MSDN
UP_ARROW = 0x26
DOWN_ARROW = 0x28
LEFT_ARROW = 0x25
RIGHT_ARROW = 0x27

L_SHIFT = 0xA0
A = 0x41
C = 0x43
X = 0x58
Z = 0x5A
F = 0x46


def Look_up():
    PressKey(UP_ARROW)
    time.sleep(0.1)
    ReleaseKey(UP_ARROW)


def restart():
    time.sleep(2.5)
    Look_up()
    time.sleep(3)
    Look_up()
    time.sleep(0.7)
    PressKey(Z)
    time.sleep(0.1)
    ReleaseKey(Z)
    time.sleep(2)


def take_action(action):
    if action['lr'] == 0:
        ReleaseKey(RIGHT_ARROW)
        PressKey(LEFT_ARROW)
    elif action['lr'] == 2:
        ReleaseKey(LEFT_ARROW)
        PressKey(RIGHT_ARROW)
    else:
        ReleaseKey(LEFT_ARROW)
        ReleaseKey(RIGHT_ARROW)

    if action['ud'] > 0.5:
        ReleaseKey(UP_ARROW)
        PressKey(DOWN_ARROW)
    elif action['ud'] < 0.5:
        ReleaseKey(DOWN_ARROW)
        PressKey(UP_ARROW)
    else:
        ReleaseKey(DOWN_ARROW)
        ReleaseKey(UP_ARROW)
    PressKey(Z) if action['Z'] > 0 else ReleaseKey(Z)
    PressKey(X) if action['X'] > 0 else ReleaseKey(X)
    PressKey(C) if action['C'] > 0 else ReleaseKey(C)
    PressKey(F) if action['F'] > 0 else ReleaseKey(F)


def ReleaseAll():
    ReleaseKey(UP_ARROW)
    ReleaseKey(DOWN_ARROW)
    ReleaseKey(LEFT_ARROW)
    ReleaseKey(RIGHT_ARROW)
    ReleaseKey(Z)
    ReleaseKey(X)
    ReleaseKey(C)
    ReleaseKey(F)


if __name__ == "__main__":
    time.sleep(1)
    while True:
        PressKey(X)
        ReleaseKey(LEFT_ARROW)
        time.sleep(0.2)
