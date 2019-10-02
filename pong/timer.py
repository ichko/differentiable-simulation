import time


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc_info):
        self.elapsed = time.time() - self.start

    def __float__(self):
        return self.elapsed


timer = Timer()
