'''Collection of generally useful utility code from the cookbook'''

from operator import itemgetter as _itemgetter
from keyword import iskeyword as _iskeyword
import itertools
import sys as _sys

#The next few are lifted from Matplotlib
def iterable(obj):
    'return true if *obj* is iterable'
    try:
        len(obj)
    except:
        return False
    return True

def iterable(obj):
    'return true if *obj* is iterable'
    try: len(obj)
    except: return False
    return True

def is_string_like(obj):
    'Return True if *obj* looks like a string'
    if isinstance(obj, (str, unicode)):
        return True
    # numpy strings are subclass of str, ma strings are not
    if ma.isMaskedArray(obj):
        if obj.ndim == 0 and obj.dtype.kind in 'SU':
            return True
        else:
            return False
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True

class Bunch:
    """
    Often we want to just collect a bunch of stuff together, naming each
    item of the bunch; a dictionary's OK for that, but a small do- nothing
    class is even handier, and prettier to use.  Whenever you want to
    group a few variables:

      >>> point = Bunch(datum=2, squared=4, coord=12)
      >>> point.datum

      By: Alex Martelli
      From: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52308
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

#Taken from a cookbook recipe.  Will be available in Python 2.6
def namedtuple(typename, field_names, verbose=False):
    """Returns a new subclass of tuple with named fields.

    >>> Point = namedtuple('Point', 'x y')
    >>> Point.__doc__                   # docstring for the new class
    'Point(x, y)'
    >>> p = Point(11, y=22)             # instantiate with positional args or keywords
    >>> p[0] + p[1]                     # indexable like a plain tuple
    33
    >>> x, y = p                        # unpack like a regular tuple
    >>> x, y
    (11, 22)
    >>> p.x + p.y                       # fields also accessable by name
    33
    >>> d = p._asdict()                 # convert to a dictionary
    >>> d['x']
    11
    >>> Point(**d)                      # convert from a dictionary
    Point(x=11, y=22)
    >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
    Point(x=100, y=22)

    """

    # Parse and validate the field names.  Validation serves two purposes,
    # generating informative error messages and preventing template injection attacks.
    if isinstance(field_names, basestring):
        field_names = field_names.replace(',', ' ').split() # names separated by whitespace and/or commas
    field_names = tuple(field_names)
    for name in (typename,) + field_names:
        if not min(c.isalnum() or c=='_' for c in name):
            raise ValueError('Type names and field names can only contain alphanumeric characters and underscores: %r' % name)
        if _iskeyword(name):
            raise ValueError('Type names and field names cannot be a keyword: %r' % name)
        if name[0].isdigit():
            raise ValueError('Type names and field names cannot start with a number: %r' % name)
    seen_names = set()
    for name in field_names:
        if name.startswith('_'):
            raise ValueError('Field names cannot start with an underscore: %r' % name)
        if name in seen_names:
            raise ValueError('Encountered duplicate field name: %r' % name)
        seen_names.add(name)

    # Create and fill-in the class template
    numfields = len(field_names)
    argtxt = repr(field_names).replace("'", "")[1:-1]   # tuple repr without parens or quotes
    reprtxt = ', '.join('%s=%%r' % name for name in field_names)
    dicttxt = ', '.join('%r: t[%d]' % (name, pos) for pos, name in enumerate(field_names))
    template = '''class %(typename)s(tuple):
        '%(typename)s(%(argtxt)s)' \n
        __slots__ = () \n
        _fields = %(field_names)r \n
        def __new__(cls, %(argtxt)s):
            return tuple.__new__(cls, (%(argtxt)s)) \n
        @classmethod
        def _make(cls, iterable, new=tuple.__new__, len=len):
            'Make a new %(typename)s object from a sequence or iterable'
            result = new(cls, iterable)
            if len(result) != %(numfields)d:
                raise TypeError('Expected %(numfields)d arguments, got %%d' %% len(result))
            return result \n
        def __repr__(self):
            return '%(typename)s(%(reprtxt)s)' %% self \n
        def _asdict(t):
            'Return a new dict which maps field names to their values'
            return {%(dicttxt)s} \n
        def _replace(self, **kwds):
            'Return a new %(typename)s object replacing specified fields with new values'
            result = self._make(map(kwds.pop, %(field_names)r, self))
            if kwds:
                raise ValueError('Got unexpected field names: %%r' %% kwds.keys())
            return result \n\n''' % locals()
    for i, name in enumerate(field_names):
        template += '        %s = property(itemgetter(%d))\n' % (name, i)
    if verbose:
        print template

    # Execute the template string in a temporary namespace
    namespace = dict(itemgetter=_itemgetter)
    try:
        exec template in namespace
    except SyntaxError, e:
        raise SyntaxError(e.message + ':\n' + template)
    result = namespace[typename]

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example).
    if hasattr(_sys, '_getframe'):
        result.__module__ = _sys._getframe(1).f_globals['__name__']

    return result

# A Least Recently Used (LRU) cache implementation

from collections import deque

def lru_cache(maxsize):
    '''Decorator applying a least-recently-used cache with the given maximum size.

    Arguments to the cached function must be hashable.
    Cache performance statistics stored in f.hits and f.misses.
    '''
    def decorating_function(f):
        cache = {}              # mapping of args to results
        queue = deque()         # order that keys have been accessed
        refcount = {}           # number of times each key is in the access queue
        def wrapper(*args):
            
            # localize variable access (ugly but fast)
            _cache=cache; _len=len; _refcount=refcount; _maxsize=maxsize
            queue_append=queue.append; queue_popleft = queue.popleft

            # get cache entry or compute if not found
            try:
                result = _cache[args]
                wrapper.hits += 1
            except KeyError:
                result = _cache[args] = f(*args)
                wrapper.misses += 1

            # record that this key was recently accessed
            queue_append(args)
            _refcount[args] = _refcount.get(args, 0) + 1

            # Purge least recently accessed cache contents
            while _len(_cache) > _maxsize:
                k = queue_popleft()
                _refcount[k] -= 1
                if not _refcount[k]:
                    del _cache[k]
                    del _refcount[k]
    
            # Periodically compact the queue by duplicate keys
            if _len(queue) > _maxsize * 4:
                for i in [None] * _len(queue):
                    k = queue_popleft()
                    if _refcount[k] == 1:
                        queue_append(k)
                    else:
                        _refcount[k] -= 1
                assert len(queue) == len(cache) == len(refcount) == sum(refcount.itervalues())

            return result
        wrapper.__doc__ = f.__doc__
        wrapper.__name__ = f.__name__
        wrapper.hits = wrapper.misses = 0
        return wrapper
    return decorating_function

#
# This can be removed once it lands in numpy trunk
#
import numpy as np

def _string_like(obj):
    try: obj + ''
    except (TypeError, ValueError): return False
    return True

def str2bool(value):
    """
    Tries to transform a string supposed to represent a boolean to a boolean.
    
    Raises
    ------
    ValueError
        If the string is not 'True' or 'False' (case independent)
    """
    value = value.upper()
    if value == 'TRUE':
        return True
    elif value == 'FALSE':
        return False
    else:
        raise ValueError("Invalid boolean")

class StringConverter(object):
    """
    Factory class for function transforming a string into another object (int,
    float).

    After initialization, an instance can be called to transform a string 
    into another object. If the string is recognized as representing a missing
    value, a default value is returned.

    Parameters
    ----------
    dtype : dtype, optional
        Input data type, used to define a basic function and a default value
        for missing data. For example, when `dtype` is float, the :attr:`func`
        attribute is set to ``float`` and the default value to `np.nan`.
    missing_values : sequence, optional
        Sequence of strings indicating a missing value.

    Attributes
    ----------
    func : function
        Function used for the conversion
    default : var
        Default value to return when the input corresponds to a missing value.
    mapper : sequence of tuples
        Sequence of tuples (function, default value) to evaluate in order.

    """
    from numpy.core import nan # To avoid circular import
    mapper = [(str2bool, None),
              (int, -1), #Needs to be int so that it can fail and promote
                         #to float
              (float, nan),
              (complex, nan+0j),
              (str, '???')]

    def __init__(self, dtype=None, missing_values=None):
        self._locked = False
        if dtype is None:
            self.func = str2bool
            self.default = None
            self._status = 0
        else:
            dtype = np.dtype(dtype).type
            if issubclass(dtype, np.bool_):
                (self.func, self.default, self._status) = (str2bool, 0, 0)
            elif issubclass(dtype, np.integer):
                #Needs to be int(float(x)) so that floating point values will
                #be coerced to int when specifid by dtype
                (self.func, self.default, self._status) = (lambda x: int(float(x)), -1, 1)
            elif issubclass(dtype, np.floating):
                (self.func, self.default, self._status) = (float, np.nan, 2)
            elif issubclass(dtype, np.complex):
                (self.func, self.default, self._status) = (complex, np.nan + 0j, 3)
            else:
                (self.func, self.default, self._status) = (str, '???', -1)

        # Store the list of strings corresponding to missing values.
        if missing_values is None:
            self.missing_values = []
        else:
            self.missing_values = set(list(missing_values) + [''])

    def __call__(self, value):
        if value in self.missing_values:
            return self.default
        return self.func(value)

    def upgrade(self, value):
        """
    Tries to find the best converter for `value`, by testing different
    converters in order.
    The order in which the converters are tested is read from the
    :attr:`_status` attribute of the instance.
        """
        try:
            self.__call__(value)
        except ValueError:
            if self._locked:
                raise
            _statusmax = len(self.mapper)
            if self._status == _statusmax:
                raise ValueError("Could not find a valid conversion function")
            elif self._status < _statusmax - 1:
                self._status += 1
            (self.func, self.default) = self.mapper[self._status]
            self.upgrade(value)

    def update(self, func, default=None, locked=False):
        """
    Sets the :attr:`func` and :attr:`default` attributes directly.

    Parameters
    ----------
    func : function
        Conversion function.
    default : var, optional
        Default value to return when a missing value is encountered.
    locked : bool, optional
        Whether this should lock in the function so that no upgrading is
        possible.
        """
        self.func = func
        self.default = default
        self._locked = locked

def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False, names=None,
            fill_empty=False):
    """
    Load data from a text file.

    Each row in the text file must have the same number of values.

    Parameters
    ----------
    fname : file or string
        File or filename to read.  If the filename extension is ``.gz``,
        the file is first decompressed.
    dtype : data-type or None, optional
        Data type of the resulting array.  If this is a record data-type,
        the resulting array will be 1-dimensional, and each row will be
        interpreted as an element of the array.   In this case, the number
        of columns used must match the number of fields in the data-type.
        If None, the dtypes will be determined by the contents of each
        column, individually.
    comments : string, optional
        The character used to indicate the start of a comment.
    delimiter : string, optional
        The string used to separate values.  By default, this is any
        whitespace.
    converters : {}
        A dictionary mapping column number or name to a function that
        will convert that column to a float.  E.g., if column 0 is a
        date string: ``converters = {0: datestr2num}``. Converters can
        also be used to provide a default value for missing data:
        ``converters = {3: lambda s: float(s or 0)}``.
    skiprows : int, optional
        Skip the first `skiprows` lines.
    usecols : sequence, optional
        Which columns to read.  This can be a sequence of either column
        numbers or column names.  For column numbers, 0 is the first.
        For example, ``usecols = (1,4,5)`` will extract the 2nd, 5th and
        6th columns.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``
    names : sequence or True, optional
        If True, the names are read from the first line after skipping
        `skiprows` lines.  If a sequence, *names* is a list of names to
        use in creating a flexible dtype for the data.

    Returns
    -------
    out : ndarray
        Data read from the text file.

    See Also
    --------
    scipy.io.loadmat : reads Matlab(R) data files

    Examples
    --------
    >>> from StringIO import StringIO   # StringIO behaves like a file object
    >>> c = StringIO("0 1\\n2 3")
    >>> np.loadtxt(c)
    array([[ 0.,  1.],
           [ 2.,  3.]])    # Make sure we're dealing with a proper dtype


    >>> d = StringIO("M 21 72\\nF 35 58")
    >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
    ...                      'formats': ('S1', 'i4', 'f4')})
    array([('M', 21, 72.0), ('F', 35, 58.0)],
          dtype=[('gender', '|S1'), ('age', '<i4'), ('weight', '<f4')])

    >>> c = StringIO("1,0,2\\n3,0,4")
    >>> x,y = np.loadtxt(c, delimiter=',', usecols=(0,2), unpack=True)
    >>> x
    array([ 1.,  3.])
    >>> y
    array([ 2.,  4.])

    """
    user_converters = converters

    if usecols is not None:
        usecols = list(usecols)

    if fill_empty:
        missing = ''
    else:
        missing = None

    if _string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname)
        else:
            fh = file(fname)
    elif hasattr(fname, 'readline'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')

    def flatten_dtype(dt):
        """Unpack a structured data-type."""
        if dt.names is None:
            return [dt]
        else:
            types = []
            for field in dt.names:
                tp, bytes = dt.fields[field]
                flat_dt = flatten_dtype(tp)
                types.extend(flat_dt)
            return types

    def split_line(line):
        """Chop off comments, strip, and split at delimiter."""
        line = line.split(comments)[0].strip()
        if line:
            return line.split(delimiter)
        else:
            return []

    # Skip the first `skiprows` lines
    for i in xrange(skiprows):
        fh.readline()

    # Read until we find a line with some values, and use
    # it to estimate the number of columns, N.
    first_vals = None
    while not first_vals:
        first_line = fh.readline()
        if first_line == '': # EOF reached
            raise IOError('End-of-file reached before encountering data.')
        first_vals = split_line(first_line)
    N = len(usecols or first_vals)

    # If names is True, read the field names from the first line
    if names == True:
        names = first_vals
        first_line = ''

    if dtype is None:
        # If we're automatically figuring out the dtype, we start with just
        # a collection of generic converters
        converters = [StringConverter(missing_values=missing)
            for i in xrange(N)]
    else:
        # Make sure we're dealing with a proper dtype
        dtype = np.dtype(dtype)
        dtype_types = flatten_dtype(dtype)
        if len(dtype_types) > 1:
            # We're dealing with a structured array, each field of
            # the dtype matches a column
            converters = [StringConverter(dt, missing) for dt in dtype_types]
        else:
            # All fields have the same dtype
            converters = [StringConverter(dtype, missing) for i in xrange(N)]

    # If usecols contains a list of names, convert them to column indices
    if usecols and _string_like(usecols[0]):
        usecols = [names.index(_) for _ in usecols]

    # By preference, use the converters specified by the user
    for i, conv in (user_converters or {}).iteritems():
        # If the converter is specified by column number, convert it to an index
        if _string_like(i):
            i = names.index(i)
        if usecols:
            try:
                i = usecols.index(i)
            except ValueError:
                # Unused converter specified
                continue
        converters[i].update(conv, None, locked=True)

    # Parse each line, including the first
    rows = []
    for i, line in enumerate(itertools.chain([first_line], fh)):
        vals = split_line(line)
        if len(vals) == 0:
            continue

        if usecols:
            vals = [vals[_] for _ in usecols]

        # If detecting dtype, see if the current converter works for this line
        if dtype is None:
            for converter, item in zip(converters, vals):
                if len(item.strip()):
                    converter.upgrade(item)

        # Store the values
        rows.append(tuple(vals))

    # Convert each value according to its column and store
    for i,vals in enumerate(rows):
        rows[i] = tuple([conv(val) for (conv, val) in zip(converters, vals)])

    # Construct final dtype if necessary
    if dtype is None:
        dtype_types = [np.array([row[i] for row in rows]).dtype
            for i in xrange(N)]
        uniform_dtype = all([dtype_types[0] == dt for dt in dtype_types])
        if uniform_dtype and not names:
            dtype = dtype_types[0]
            dtype_types = dtype
        else:
            if not names:
                names = ['column%d'%i for i in xrange(N)]
            elif usecols:
                names = [names[i] for i in usecols]
            dtype = np.dtype(zip(names, dtype_types))
    else:
        # Override the names if specified
        if dtype.names and names:
            dtype.names = names

    if len(dtype_types) > 1:
        # We're dealing with a structured array, with a dtype such as
        # [('x', int), ('y', [('s', int), ('t', float)])]
        #
        # First, create the array using a flattened dtype:
        # [('x', int), ('s', int), ('t', float)]
        #
        # Then, view the array using the specified dtype.
        rows = np.array(rows, dtype=np.dtype([('', t) for t in dtype_types]))
        rows = rows.view(dtype)
    else:
        rows = np.array(rows, dtype)

    rows = np.squeeze(rows)
    if unpack:
        return rows.T
    else:
        return rows

#Taken from a numpy-discussion mailing list post 'Re: adding field to rec array'
#by Robert Kern
def append_field(rec, name, arr, dtype=None):
    """
    Appends a field to an existing record array.

    Parameters
    ----------
    rec : numpy record array
        Array to which the new field should be appended
    name : string
        Name to be given to the new field
    arr : ndarray
        Array containing the data for the new field.
    dtype : data-type or None, optional
        Data type of the new field.  If this is None, the data type will
        be obtained from `arr`.

    Returns
    -------
    out : numpy record array
        `rec` with the new field appended.
    rec = np.asarray(rec, name, arr)
    """
    if not iterable(name):
        name = [name]
    if not iterable(arr)
        arr = [arr]

    if dtype is None:
        dtype = [a.dtype for a in arr]

    newdtype = np.dtype(rec.dtype.descr + zip(name, dtype))
    newrec = np.empty(rec.shape, dtype=newdtype)
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
    for n,a in zip(name, arr):
        newrec[n] = a
    return newrec
