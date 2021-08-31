# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Solver to automatically calculate derived parameters from a dataset."""

from collections import ChainMap, deque
import contextlib
from inspect import signature, Parameter


class Path:
    def __init__(self, steps, have, need):
        self.steps = steps
        self.have = have
        self.need = need

    def is_complete(self):
        return not bool(self.need)

    def __add__(self, other):
        if any(f in set(self.steps) for f in other.steps):
            raise ValueError(f'{other.steps} already in steps')

        # Prepend steps so that final path is in proper call order
        # Don't really "have" what's in the new function call, but instead it just needs
        # to be removed from what's needed.
        return Path(other.steps + self.steps, self.have,
                    (self.need | other.need) - (self.have | other.have))

    def __str__(self):
        return (f'Path<Steps: {[f.__name__ for f in self.steps]} Have: {self.have} '
                f'Need: {self.need}>')

    __repr__ = __str__


class Solver:
    names = {'tw': 'wet_bulb_temperature', 'td': 'dewpoint_temperature',
             'dewpoint': 'dewpoint_temperature', 'tv': 'virtual_temperature',
             'q': 'specific_humidity', 'r': 'mixing_ratio', 'rh': 'relative_humidity',
             'p': 'pressure', 't': 'temperature', 'isobaric': 'pressure'}

    standard_names = {'temperature': 'air_temperature'}

    fallback_names = {'temperature': ['temp'], 'pressure': ['P', 'isobaric']}

    def __init__(self):
        self._graph = {}
        self._funcs = {}

    def register(self, *args, inputs=None):
        def dec(func):
            nonlocal inputs
            nonlocal args
            if inputs is None:
                funcsig = signature(func)
                inputs = [name for name, param in funcsig.parameters.items() if
                          param.default is Parameter.empty]

            if not args:
                args = (func.__name__,)

            normed_returns = self.normalize_names(args)
            normed_inputs = self.normalize_names(inputs)
            path = Path([func], set(normed_returns), set(normed_inputs))
            self._funcs[func] = (normed_inputs, normed_returns)
            for ret in normed_returns:
                self._graph.setdefault(ret, []).append(path)
            return func

        return dec

    def normalize_names(self, names):
        return [self.normalize(name) for name in names]

    def normalize(self, name):
        return self.names.get(name.lower(), name.lower())

    def _map_func_args(self, func, data):
        key_map = {self.normalize(key): key for key in ChainMap(data, data.coords)}
        for item in self._funcs[func][0]:
            if item in key_map:
                yield data[key_map[item]]
            elif item in self.standard_names:
                ds = data.filter_by_attrs(standard_name=self.standard_names[item])
                yield next(iter(ds))
            else:
                for name in self.fallback_names.get(item, []):
                    if name in key_map:
                        yield data[key_map[name]]

    def calculate(self, data, name):
        data = data.copy()
        for func in self.solve(set(data) | set(data.coords), name):
            result = func(*self._map_func_args(func, data))
            retname = self._funcs[func][-1]
            if isinstance(result, tuple):
                for name, val in zip(retname, result):
                    data[name] = val
            else:
                data[retname] = result

        return data[self.normalize(name)]

    def solve(self, have, want):
        # Using deque as a FIFO queue by pushing at one end and popping
        # from the other--this makes this a Breadth-First Search
        options = deque([Path([], set(self.normalize_names(have)), {self.normalize(want)})])
        while options:
            path = options.popleft()
            # If calculation path is complete, return the steps
            if path.is_complete():
                return path.steps
            else:
                # Otherwise grab one of the remaining needs and
                # add all methods for calculating to the current steps
                # and make them options to consider
                item = path.need.pop()
                for trial_step in self._graph.get(item, ()):
                    # ValueError gets thrown if we try to repeat a function call
                    with contextlib.suppress(ValueError):
                        options.append(path + trial_step)

        raise ValueError(f'Unable to calculate {want} from {have}')


solver = Solver()
