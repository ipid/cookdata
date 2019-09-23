__all__ = ('TimingFit', 'time_it')

import time


class TimingFit:
    __slots__ = ('_classifier', '_do_print', 'time_cost')

    def __init__(self, classifier, print_time=True):
        self._classifier = classifier
        self._do_print = print_time
        self.time_cost = .0

    def fit(self, X, y, print_time=None, **kwargs):
        time_begin = time.perf_counter()
        result = self._classifier.fit(X, y, **kwargs)
        time_cost = time.perf_counter() - time_begin
        self.time_cost = time_cost

        if print_time is None:
            print_time = self._do_print
        if print_time:
            print(f'\n[fit executed for: {time_cost:.6f}s]\n')

        return result


def time_it(func, fn_name='Function'):
    def wrapper(*args, **kwargs):
        time_begin = time.perf_counter()
        result = func(*args, **kwargs)
        time_cost = round(time.perf_counter() - time_begin, 6)
        print(f'\n[{fn_name} executed for: {time_cost}s]\n')
        return result

    return wrapper
