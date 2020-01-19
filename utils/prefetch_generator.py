import concurrent.futures


class PrefetchGenerator:
    def __init__(self, handler, num_to_prefetch, max_workers):
        self.Q = []
        self.I = 0
        self.handler = handler
        self.num_to_prefetch = num_to_prefetch
        self.max_workers = max_workers
        self.client = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers)

    def shutdown(self):
        self.client.shutdown()
        self.client = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers)
        self.Q = []

    def __iter__(self):
        while True:
            for _ in range(self.num_to_prefetch - len(self.Q)):
                future = self.client.submit(self.handler, self.I)
                self.Q.append(future)
                self.I += 1

            for future in self.Q:
                if not future.running():
                    value = future.result()
                    yield value
                    self.Q.remove(future)
                    break


def prefetch(num_to_prefetch=32, max_workers=None):
    def decorated(handler):
        return lambda: PrefetchGenerator(handler, num_to_prefetch, max_workers)

    return decorated


if __name__ == '__main__':
    import numpy as np

    A = np.random.rand(100, 10, 10, 3)
    BS = 3

    @prefetch(num_to_prefetch=32, max_workers=None)
    def image_generator(I):
        l, r = I * BS, I * BS + BS
        return A[l:r]

    i = iter(image_generator)
    batch = next(i)
    print(batch.shape)
