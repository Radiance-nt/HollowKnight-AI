import threading
import time
import collections
import cv2
import torch
import win32gui, win32ui, win32con, win32api
import numpy as np


class FrameBuffer():
    def __init__(self):
        self._stop_event = threading.Event()
        self.hwnd = win32gui.FindWindow(None, 'Hollow Knight')
        rect = win32gui.GetWindowRect(self.hwnd)

        self.left, self.top, x2, y2 = rect
        self.width = rect[2] - rect[0]
        self.height = rect[3] - rect[1]

        self.hwindc = win32gui.GetWindowDC(self.hwnd)
        self.srcdc = win32ui.CreateDCFromHandle(self.hwindc)
        self.memdc = self.srcdc.CreateCompatibleDC()
        self.bmp = win32ui.CreateBitmap()
        self.bmp.CreateCompatibleBitmap(self.srcdc, self.width, self.height)

    def get_frame(self):
        self.memdc.SelectObject(self.bmp)
        self.memdc.BitBlt((0, 0), (self.width, self.height), self.srcdc, (0, 0), win32con.SRCCOPY)
        signedIntsArray = self.bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.height, self.width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (160, 80))
        cv2.imshow('frame', img)
        cv2.waitKey(1)
        # if not isinstance(img, torch.Tensor):
        #     if len(img.shape) < 3:
        #         img = torch.Tensor(img).unsqueeze(0)
        #     img = img.unsqueeze(0)
        return img
