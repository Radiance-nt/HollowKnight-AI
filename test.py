from threading import Thread
import time

import torch.multiprocessing as mp


class P:
    def __init__(self):
        self.i = 1


def printa(P):
    while True:
        print(P.i)


if __name__ == '__main__':

    Pi = P()
    warm_process = Thread(target=printa, args=(Pi,),daemon=True)
    warm_process.start()

    time.sleep(1)

    Pi.i = 2
    time.sleep(3)
