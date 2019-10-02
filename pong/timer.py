import time


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc_info):
        self.elapsed = time.time() - self.start

    def __float__(self):
        return self.elapsed


class PrintTimer(Timer):
    def __init__(self, format='%.2fs'):
        self.format = format

    def __exit__(self, *exc_info):
        super(PrintTimer, self).__exit__(None, None)
        print(self.format % self.elapsed)


print_timer = lambda *args: PrintTimer(*args)
timer = Timer()
