'Collection of tools for managing the package'

# Used to specify functions that should be exported--i.e. added to __all__
# Inspired by David Beazley and taken from python-ideas:
# https://mail.python.org/pipermail/python-ideas/2014-May/027824.html

__all__ = ['Exporter']


class Exporter(object):
    def __init__(self, globls):
        self.globls = globls
        self.exports = globls.setdefault('__all__', [])

    def export(self, defn):
        self.exports.append(defn.__name__)
        return defn

    def __enter__(self):
        self.cur_vars = list(self.globls.keys())

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k in self.globls.keys():
            if k not in self.cur_vars:
                self.exports.append(k)
