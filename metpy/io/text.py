import re
from datetime import datetime

from ..cbook import is_string_like


# Generic error for problems parsing text reports
class ParseError(Exception):
    pass


def parse_wmo_time(s, ref_time=None):
    r'''Parse WMO-formatted time string (DDHHMMZ)

    Turns the WMO string into a full `datetime.datetime` instance, using the
    reference time to fill in the missing parts.

    Parameters
    ----------
    s : string
        The source WMO-formatted date/time string

    ref_time : `datetime.datetime` or None
        The reference time to use. If None, the current UTC time is used.

    Returns
    -------
        The corresponding `datetime.datetime` instance
    '''
    if not ref_time:
        ref_time = datetime.utcnow()
    return datetime.strptime(s, '%d%H%MZ').replace(year=ref_time.year,
                                                   month=ref_time.month)


class StringIter(object):
    def __init__(self, text, skip_blank=False):
        self._buffer = text
        self.skip_blank = skip_blank
        self._cursor = 0

    def __iter__(self):
        return self


class LineIter(StringIter):
    r'''Iterates over lines from a buffer using an arbitrary delimiter.'''
    def __init__(self, text, skip_blank=False, linesep='\n\n'):
        super(LineIter, self).__init__(text, skip_blank)
        self.linesep = linesep

    @property
    def linesep(self):
        return self._linesep

    @linesep.setter
    def linesep(self, pattern):
        self._linesep = re.compile(pattern)

    def _find(self):
        match = self._linesep.search(self._buffer, self._cursor)
        if match is None:
            return None
        else:
            return match.start(), match.end()

    def peek(self):
        r'Get the next item without advancing any pointers.'
        rng = self._find()
        if rng is None:
            return ''
        else:
            return self._buffer[self._cursor:rng[0]]

    def __next__(self):
        # Look for the next separator. If not found, raise StopIteration
        rng = self._find()
        if rng is None:
            raise StopIteration()

        # Get the next slice
        rng_start, rng_end = rng
        ret = self._buffer[self._cursor:rng_start]

        # Advance the cursor to end of match
        self._cursor = rng_end

        # Check if we should skip because blank--if so, just recurse
        if self.skip_blank and not ret:
            return next(self)
        else:
            return ret


class ProductIter(StringIter):
    r'''Iterates over products from a buffer based on start and end markers.'''
    def __init__(self, text, skip_blank=False, start_marker=chr(0x1), end_marker=chr(0x3)):
        super(ProductIter, self).__init__(text, skip_blank)
        self.bom = start_marker
        self.eom = end_marker

    def __next__(self):
        # Look for the next start. If not found, raise StopIteration
        start = self._buffer.find(self.bom, self._cursor)
        if start == -1:
            raise StopIteration()

        # Find the end
        self._cursor = self._buffer.find(self.eom, start)

        # Return the slice, making sure we skip the BOM
        return self._buffer[start + 1:self._cursor]


class TextProductFile(object):
    r'''Allows iterating over products from a file rather than lines.'''
    def __init__(self, f):
        if is_string_like(f):
            # Like ascii, but forgiving of characters in [128, 255]
            self._fobj = open(f, 'rt', encoding='latin-1')
        else:
            self._fobj = f

        self._buffer = self._fobj.read()

    def __iter__(self):
        return ProductIter(self._buffer)


class RegexParser(object):
    r'''Helper for parsing. Takes a regex and either returns the match, or hands the
    named groups to a post-processor.'''
    def __init__(self, pattern, postprocess=None, default=None, repeat=False, keepall=True):
        self._regex = re.compile(pattern, re.VERBOSE)
        self._post = postprocess
        self._default = default
        self.repeat = repeat
        self.keepall = keepall

    def parse(self, string, start=0, *context):
        match = self._regex.search(string, start)
        if match:
            if self._post:
                res = self._post(match.groupdict(), *context)
                return (match.start(), match.end()), res
            else:
                return (match.start(), match.end()), match.group()

        return None, self._default


class WMOTextProduct(object):
    r'''Parses a WMO-formatted text product.'''
    def __init__(self, text, ref_time=None):
        self.ref_time = ref_time
        line_iter = LineIter(text, skip_blank=True)
        self._parse_header(line_iter)
        self._parse(line_iter)

    def _parse_header(self, it):
        self.seq_num = int(next(it))
        parts = next(it).split(' ')
        self.data_designator = parts[0]
        self.center = parts[1]
        self.datetime = parse_wmo_time(parts[2] + 'Z', self.ref_time)
        if len(parts) > 3:
            self.additional = parts[3]
        else:
            self.additional = ''

    def _parse(self, it):
        pass

    def __str__(self):
        fmt = ['Sequence Number: {0.seq_num}', 'Data Designator: {0.data_designator}',
               'Center: {0.center}', 'Date/Time: {0.datetime}']
        if self.additional:
            fmt.append('Additional: {0.additional}')
        return '\n\t'.join(fmt).format(self)
