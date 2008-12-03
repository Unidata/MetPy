'''Collection of generally useful utility code from the cookbook'''

from operator import itemgetter as _itemgetter
from keyword import iskeyword as _iskeyword
import itertools
import sys as _sys
from numpy import ma

#The next few are lifted from Matplotlib
def iterable(obj):
    'return true if *obj* is iterable'
    try:
        len(obj)
    except:
        return False
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
import numpy.ma as ma

def _to_filehandle(fname, flag='r', return_opened=False):
    """
    Returns the filehandle corresponding to a string or a file.
    If the string ends in '.gz', the file is automatically unzipped.
    
    Parameters
    ----------
    fname : string, filehandle
        Name of the file whose filehandle must be returned.
    flag : string, optional
        Flag indicating the status of the file ('r' for read, 'w' for write).
    return_opened : boolean, optional
        Whether to return the opening status of the file.
    """
    if is_string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fhd = gzip.open(fname, flag)
        elif fname.endswith('.bz2'):
            import bz2
            fhd = bz2.BZ2File(fname)
        else:
            fhd = file(fname, flag)
        opened = True
    elif hasattr(fname, 'seek'):
        fhd = fname
        opened = False
    else:
        raise ValueError('fname must be a string or file handle')
    if return_opened:
        return fhd, opened
    return fhd


def flatten_dtype(ndtype):
    """
    Unpack a structured data-type.

    """
    names = ndtype.names
    if names is None:
        return [ndtype]
    else:
        types = []
        for field in names:
            (typ, _) = ndtype.fields[field]
            flat_dt = flatten_dtype(typ)
            types.extend(flat_dt)
        return types


def nested_masktype(datatype):
    """
    Construct the dtype of a mask for nested elements.

    """
    names = datatype.names
    if names:
        descr = []
        for name in names:
            (ndtype, _) = datatype.fields[name]
            descr.append((name, nested_masktype(ndtype)))
        return descr
    # Is this some kind of composite a la (np.float,2)
    elif datatype.subdtype:
        mdescr = list(datatype.subdtype)
        mdescr[0] = np.dtype(bool)
        return tuple(mdescr)
    else:
        return np.bool


class LineSplitter:
    """
    Defines a function to split a string at a given delimiter or at given places.
    
    Parameters
    ----------
    comment : {'#', string}
        Character used to mark the beginning of a comment.
    delimiter : var
        
    """
    def __init__(self, delimiter=None, comments='#'):
        self.comments = comments
        # Delimiter is a character
        if delimiter is None:
            self._isfixed = False
            self.delimiter = None
        elif is_string_like(delimiter):
            self._isfixed = False
            self.delimiter = delimiter.strip() or None
        # Delimiter is a list of field widths
        elif hasattr(delimiter, '__iter__'):
            self._isfixed = True
            idx = np.cumsum([0]+list(delimiter))
            self.slices = [slice(i,j) for (i,j) in zip(idx[:-1], idx[1:])]
        # Delimiter is a single integer
        elif int(delimiter):
            self._isfixed = True
            self.slices = None
            self.delimiter = delimiter
        else:
            self._isfixed = False
            self.delimiter = None
    #
    def __call__(self, line):
        # Strip the comments
        line = line.split(self.comments)[0]
        if not line:
            return []
        # Fixed-width fields
        if self._isfixed:
            # Fields have different widths
            if self.slices is None:
                fixed = self.delimiter
                slices = [slice(i, i+fixed)
                          for i in range(len(line))[::fixed]]
            else:
                slices = self.slices
            return [line[s].strip() for s in slices]
        else:
            return [s.strip() for s in line.split(self.delimiter)]

        """
    Splits the line at each current delimiter.
    Comments are stripped beforehand.
        """


class NameValidator:
    """
    Validates a list of strings to use as field names.
    The strings are stripped of any non alphanumeric character, and spaces
    are replaced by `_`.

    During instantiation, the user can define a list of names to exclude, as 
    well as a list of invalid characters. Names in the exclude list are appended
    a '_' character.

    Once an instance has been created, it can be called with a list of names
    and a list of valid names will be created.
    The `__call__` method accepts an optional keyword, `default`, that sets
    the default name in case of ambiguity. By default, `default = 'f'`, so
    that names will default to `f0`, `f1`

    Parameters
    ----------
    excludelist : sequence, optional
        A list of names to exclude. This list is appended to the default list
        ['return','file','print'].
    deletechars : string, optional
        A string combining invalid characters that must be deleted from the names.
    """
    #
    defaultexcludelist = ['return','file','print']
    defaultdeletechars = set("""~!@#$%^&*()-=+~\|]}[{';: /?.>,<""")
    #
    def __init__(self, excludelist=None, deletechars=None):
        #
        if excludelist is None:
            excludelist = []
        excludelist.append(self.defaultexcludelist)
        self.excludelist = excludelist
        #
        if deletechars is None:
            delete = self.defaultdeletechars
        else:
            delete = set(deletechars)
        delete.add('"')
        self.deletechars = delete
    #
    def validate(self, names, default='f'):
        #
        if names is None:
            return
        #
        validatednames = []
        seen = dict()
        #
        deletechars = self.deletechars
        excludelist = self.excludelist
        for i, item in enumerate(names):
            item = item.strip().replace(' ', '_')
            item = ''.join([c for c in item if c not in deletechars])
            if not len(item):
                item = '%s%d' % (default, i)
            elif item in excludelist:
                item += '_'
            cnt = seen.get(item, 0)
            if cnt > 0:
                validatednames.append(item + '_%d' % cnt)
            else:
                validatednames.append(item)
            seen[item] = cnt+1
        return validatednames
    #
    def __call__(self, names, default='f'):
        return self.validate(names, default)



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



class StringConverter:
    """
    Factory class for function transforming a string into another object (int,
    float).

    After initialization, an instance can be called to transform a string 
    into another object. If the string is recognized as representing a missing
    value, a default value is returned.

    Parameters
    ----------
    dtype_or_func : {None, dtype, function}, optional
        Input data type, used to define a basic function and a default value
        for missing data. For example, when `dtype` is float, the :attr:`func`
        attribute is set to ``float`` and the default value to `np.nan`.
        Alternatively, function used to convert a string to another object.
        In that later case, it is recommended to give an associated default
        value as input.
    default : {None, var}, optional
        Value to return by default, that is, when the string to be converted
        is flagged as missing.
    missing_values : {sequence}, optional
        Sequence of strings indicating a missing value.
    locked : {boolean}, optional
        Whether the StringConverter should be locked to prevent automatic 
        upgrade or not.

    Attributes
    ----------
    func : function
        Function used for the conversion
    default : var
        Default value to return when the input corresponds to a missing value.
    _status : integer
        Integer representing the order of the conversion.
    _mapper : sequence of tuples
        Sequence of tuples (dtype, function, default value) to evaluate in order.
    _locked : boolean
        Whether the StringConverter is locked, thereby preventing automatic any
        upgrade or not.

    """
    #
    _mapper = [(np.bool_, str2bool, None),
               (np.integer, int, -1),
               (np.floating, float, np.nan),
               (np.complex, complex, np.nan+0j),
               (np.string_, str, '???')]
    (_defaulttype, _defaultfunc, _defaultfill) = zip(*_mapper)
    #
    @classmethod
    def _getsubdtype(cls, val):
        """Returns the type of the dtype of the input variable."""
        return np.array(val).dtype.type
    #
    @classmethod
    def upgrade_mapper(cls, func, default=None):
        """
    Upgrade the mapper of a StringConverter by adding a new function and its
    corresponding default.
    
    The input function (or sequence of functions) and its associated default 
    value (if any) is inserted in penultimate position of the mapper.
    The corresponding type is estimated from the dtype of the default value.
    
    Parameters
    ----------
    func : var
        Function, or sequence of functions
        """
        # Func is a single functions
        if hasattr(func, '__call__'):
            cls._mapper.insert(-1, (cls._getsubdtype(default), func, default))
            return
        elif hasattr(func, '__iter__'):
            if isinstance(func[0], (tuple, list)):
                for _ in func:
                    cls._mapper.insert(-1, _)
                return
            if default is None:
                default = [None] * len(func)
            else:
                default = list(default)
                default.append([None] * (len(func)-len(default)))
            for (fct, dft) in zip(func, default):
                cls._mapper.insert(-1, (cls._getsubdtype(dft), fct, dft))
    #
    def __init__(self, dtype_or_func=None, default=None, missing_values=None,
                 locked=False):
        # Defines a lock for upgrade
        self._locked = bool(locked)
        # No input dtype: minimal initialization
        if dtype_or_func is None:
            self.func = str2bool
            self._status = 0
            self.default = default
        else:
            # Is the input a np.dtype ?
            try:
                self.func = None
                ttype = np.dtype(dtype_or_func).type
            except TypeError:
                # dtype_or_func must be a function, then
                if not hasattr(dtype_or_func, '__call__'):
                    errmsg = "The input argument `dtype` is neither a function"\
                             " or a dtype (got '%s' instead)"
                    raise TypeError(errmsg % type(dtype_or_func))
                # Set the function
                self.func = dtype_or_func
                # If we don't have a default, try to guess it or set it to None
                if default is None:
                    try:
                        default = self.func('0')
                    except ValueError:
                        default = None
                ttype = self._getsubdtype(default)
            # Set the status according to the dtype
            for (i, (deftype, func, default_def)) in enumerate(self._mapper):
                if np.issubdtype(ttype, deftype):
                    self._status = i
                    self.default = default or default_def
                    break
            # If the input was a dtype, set the function to the last we saw
            if self.func is None:
                self.func = func
            # If the status is 1 (int), change the function to smthg more robust
            if self.func == self._mapper[1][1]:
                self.func = lambda x : int(float(x))
        # Store the list of strings corresponding to missing values.
        if missing_values is None:
            self.missing_values = set([''])
        else:
            self.missing_values = set(list(missing_values) + [''])
    #
    def __call__(self, value):
        try:
            return self.func(value)
        except ValueError:
            if value in self.missing_values:
                return self.default
            raise ValueError("Cannot convert string '%s'" % value)
    #
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
            # Raise an exception if we locked the converter...
            if self._locked:
                raise ValueError("Converter is locked and cannot be upgraded")
            _statusmax = len(self._mapper)
            # Complains if we try to upgrade by the maximum
            if self._status == _statusmax:
                raise ValueError("Could not find a valid conversion function")
            elif self._status < _statusmax - 1:
                self._status += 1
            (_, self.func, self.default) = self._mapper[self._status]
            self.upgrade(value)
    #
    def update(self, func, default=None, missing_values='', locked=False):
        """
    Sets the :attr:`func` and :attr:`default` attributes directly.

    Parameters
    ----------
    func : function
        Conversion function.
    default : {var}, optional
        Default value to return when a missing value is encountered.
    missing_values : {var}, optional
        Sequence of strings representing missing values.
    locked : {False, True}, optional
        Whether the status should be locked to prevent automatic upgrade.
        """
        self.func = func
        self._locked = locked
        # Don't reset the default to None if we can avoid it
        if default is not None:
            self.default = default
        # Add the missing values to the existing set
        if missing_values is not None:
            if is_string_like(missing_values):
                self.missing_values.add(missing_values)
            elif hasattr(missing_values, '__iter__'):
                for val in missing_values:
                    self.missing_values.add(val)
        else:
            self.missing_values = []


def genloadtxt(fname, dtype=float, comments='#', delimiter=None, skiprows=0,
               converters=None, missing='', missing_values=None,
               usecols=None, unpack=None,
               names=None, excludelist=None, deletechars=None):
    """
    Load data from a text file.

    Each row in the text file must have the same number of values.

    Parameters
    ----------
    fname : file or string
        File or filename to read.  If the filename extension is `.gz` or `.bz2`,
        the file is first decompressed.
    dtype : data-type
        Data type of the resulting array.  If this is a flexible data-type,
        the resulting array will be 1-dimensional, and each row will be
        interpreted as an element of the array. In this case, the number
        of columns used must match the number of fields in the data-type,
        and the names of each field will be set by the corresponding name
        of the dtype.
        If None, the dtypes will be determined by the contents of each
        column, individually.
    comments : {string}, optional
        The character used to indicate the start of a comment.
    delimiter : {string}, optional
        The string used to separate values.  By default, this is any
        whitespace.
    skiprows : {int}, optional
        Numbers of lines to skip at the beginning of the file.
    converters : {None, dictionary}, optional
        A dictionary mapping column number to a function that will convert
        that column to a float.  E.g., if column 0 is a date string:
        ``converters = {0: datestr2num}``. Converters can also be used to
        provide a default value for missing data:
        ``converters = {3: lambda s: float(s or 0)}``.
    missing : {string}, optional
        A string representing a missing value, irrespective of the column where
        it appears (e.g., `'missing'` or `'unused'`).
    missing_values : {None, dictionary}, optional
        A dictionary mapping a column number to a string indicating whether the
        corresponding field should be masked.
    usecols : {None, sequence}, optional
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
    names : {None, True, string, sequence}, optional
        If `names` is True, the field names are read from the first valid line
        after the first `skiprows` lines.
        If `names` is a sequence or a single-string of comma-separated names,
        the names will be used to define the field names in a flexible dtype.
        If `names` is None, the names of the dtype fields will be used, if any.
    unpack : {bool}, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``


    Returns
    -------
    out : MaskedArray
        Data read from the text file.

    Notes
    --------
    * When spaces are used as delimiters, there should not be any missing data
      between two fields.
    * When `names` is not None, names are lower cased, the spaces replaced by
      underscores, and any illegal character suppressed.
    * When the variable are named (either by a flexible dtype or with `names`,
      there must not be any header in the file (else a :exc:ValueError exception
      is raised).


    """
    user_converters = converters or {}
    if not isinstance(user_converters, dict):
        errmsg = "The input argument 'converter' should be a valid dictionary "\
                 "(got '%s' instead)"
        raise TypeError(errmsg % type(user_converters))
    user_missing_values = missing_values or {}
    if not isinstance(user_missing_values, dict):
        errmsg = "The input argument 'missing_values' should be a valid "\
                 "dictionary (got '%s' instead)"
        raise TypeError(errmsg % type(missing_values))
    defmissing = [_.strip() for _ in missing.split(',')] + ['']

    # Initialize the filehanlde, the LineSplitter and the NameValidator
    fhd = _to_filehandle(fname)
    split_line = LineSplitter(delimiter=delimiter, comments=comments)
    validate_names = NameValidator(excludelist=excludelist,
                                   deletechars=deletechars)

    # Get the first valid lines after the first skiprows ones
    for i in xrange(skiprows):
        fhd.readline()
    first_values = None
    while not first_values:
        first_line = fhd.readline()
        if first_line == '':
            raise IOError('End-of-file reached before encountering data.')
        first_values = split_line(first_line)

    # Check the columns to use
    if usecols is not None:
        usecols = list(usecols)
    nbcols = len(usecols or first_values)

    #Whether or not we have read the names from the file
    read_names = False

    # Check the names and overwrite the dtype.names if needed
    if dtype is not None:
        dtype = np.dtype(dtype)
    dtypenames = getattr(dtype, 'names', None)
    if names is True:
        names = validate_names(first_values)
        first_line =''
        read_names = True
    elif is_string_like(names):
        names = validate_names([_.strip() for _ in names.split(',')])
    elif names:
        names = validate_names(names)
    elif dtypenames:
        dtype.names = validate_names(dtypenames)
    if names and dtypenames:
        dtype.names = names

    # If usecols is a list of names, convert to a list of indices
    if usecols:
        for (i, current) in enumerate(usecols):
            if is_string_like(current):
                usecols[i] = names.index(current)

    # If user_missing_values has names as keys, transform them to indices
    missing_values = {}
    for (key, val) in user_missing_values.iteritems():
        # If val is a list, flatten it. In any case, add missing &'' to the list
        if isinstance(val, (list, tuple)):
            val = [str(_) for _ in val]
        else:
            val = [str(val),]
        val.extend(defmissing)
        if is_string_like(key):
            try:
                missing_values[names.index(key)] = val
            except ValueError:
                pass
        else:
            missing_values[key] = val


    # Initialize the default converters
    if dtype is None:
        # Note: we can't use a [...]*nbcols, as we would have 3 times the same
        # ... converter, instead of 3 different converters.
        converters = [StringConverter(None,
                              missing_values=missing_values.get(_, defmissing))
                      for _ in range(nbcols)]
    else:
        flatdtypes = flatten_dtype(dtype)
        # Initialize the converters
        if len(flatdtypes) > 1:
            # Flexible type : get a converter from each dtype
            converters = [StringConverter(dt,
                              missing_values=missing_values.get(i, defmissing))
                          for (i, dt) in enumerate(flatdtypes)]
        else:
            # Set to a default converter (but w/ different missing values)
            converters = [StringConverter(dtype,
                              missing_values=missing_values.get(_, defmissing))
                          for _ in range(nbcols)]
    missing_values = [_.missing_values for _ in converters]

    # Update the converters to use the user-defined ones
    for (i, conv) in user_converters.iteritems():
        # If the converter is specified by column names, use the index instead
        if is_string_like(i):
            i = names.index(i)
        if usecols:
            try:
                i = usecols.index(i)
            except ValueError:
                # Unused converter specified
                continue
        converters[i].update(conv, default=None, 
                             missing_values=missing_values[i],
                             locked=True)

    rows = []
    rowmasks = []
    # Parse each line
    for (i, line) in enumerate(itertools.chain([first_line,], fhd)):
        values = split_line(line)
        # Skip an empty line
        if len(values) == 0:
            continue
        # Select only the columns we need
        if usecols:
            values = [values[_] for _ in usecols]
        # Check whether we need to update the converter
        if dtype is None:
            for (j, (converter, item)) in enumerate(zip(converters, values)):
                if len(item.strip()):
                    converter.upgrade(item)
        # Store the values
        rows.append(tuple(values))
        rowmasks.append(tuple([val in mss 
                               for (val, mss) in zip(values, missing_values)]))

    # Convert each value according to the converter:
    for (i, vals) in enumerate(rows):
        rows[i] = tuple([conv(val) for (conv, val) in zip(converters, vals)])

    # If we read the names from the file, we have more names than columns
    # we want, so we need to filter down to the names we actually want.
    if usecols and read_names:
        names = [names[_] for _ in usecols]

    # Reset the dtype 
    if dtype is None:
        # Get the dtypes from the first row
        coldtypes = [np.array(val).dtype for val in rows[0]]
        # Find the columns with strings, and take the largest number of chars.
        strcolidx = [i for (i, v) in enumerate(coldtypes) if v.char == 'S']
        for i in strcolidx:
            coldtypes[i] = "|S%i" % max(len(row[i]) for row in rows)
        
        if names is None:
            # If the dtype is uniform, don't define names, else use ''
            base = coldtypes[0]
            if np.all([(dt == base) for dt in coldtypes]):
                (ddtype, mdtype) = (base, np.bool)
            else:
                ddtype = [('', dt) for dt in coldtypes]
                mdtype = [('', np.bool) for dt in coldtypes]
        else:
            ddtype = zip(names, coldtypes)
            mdtype = zip(names, [np.bool] * len(coldtypes))
        output = np.array(rows, dtype=ddtype)
        outputmask = np.array(rowmasks, dtype=mdtype)
    else:
        # Overwrite the initial dtype names if needed
        if names and dtype.names:
            dtype.names = names
        # Check whether we have a nested dtype
        flatdtypes = flatten_dtype(dtype)
        if len(flatdtypes) > 1:
            # Nested dtype, eg  [('a', int), ('b', [('b0', int), ('b1', 'f4')])]
            # First, create the array using a flattened dtype:
            # [('a', int), ('b1', int), ('b2', float)]
            # Then, view the array using the specified dtype.
            rows = np.array(rows, dtype=np.dtype([('', t) for t in flatdtypes]))
            output = rows.view(dtype)
            # Now, process the rowmasks the same way
            rowmasks = np.array(rowmasks, dtype=np.dtype([('', np.bool)
                                                          for t in flatdtypes]))
            # Construct the new dtype
            mdtype = nested_masktype(dtype)
            outputmask = rowmasks.view([tuple(_) for _ in mdtype])
        else:
            output = np.array(rows, dtype)
            if dtype.names:
                outputmask = np.array(rowmasks,
                                      dtype=[(_, np.bool) for _ in dtype.names])
            else:
                outputmask = np.array(rowmasks, dtype=np.bool)
    # Construct the final array
    if unpack:
        return (output.squeeze().T, outputmask.squeeze().T)
    return (output.squeeze(), outputmask.squeeze().T)


def loadtxt(fname, dtype=float, comments='#', delimiter=None, skiprows=0,
               converters=None, missing='', missing_values=None,
               usecols=None, unpack=None,
               names=None, excludelist=None, deletechars=None):
    kwargs = dict(dtype=dtype, comments=comments, delimiter=delimiter, 
                  skiprows=skiprows, converters=converters,
                  missing=missing, missing_values=missing_values,
                  usecols=usecols, unpack=unpack, names=names, 
                  excludelist=excludelist, deletechars=deletechars)
    (output, _) = genloadtxt(fname, **kwargs)
    return output

def mloadtxt(fname, dtype=float, comments='#', delimiter=None, skiprows=0,
               converters=None, missing='', missing_values=None,
               usecols=None, unpack=None,
               names=None, excludelist=None, deletechars=None):
    kwargs = dict(dtype=dtype, comments=comments, delimiter=delimiter, 
                  skiprows=skiprows, converters=converters,
                  missing=missing, missing_values=missing_values,
                  usecols=usecols, unpack=unpack, names=names, 
                  excludelist=excludelist, deletechars=deletechars)
    (output, outputmask) = genloadtxt(fname, **kwargs)
    output = output.view(ma.MaskedArray)
    output.mask = outputmask
    return output

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
    if not iterable(arr):
        arr = [arr]

    if dtype is None:
        dtype = [a.dtype for a in arr]

    newdtype = np.dtype(rec.dtype.descr + zip(name, dtype))
    newrec = np.empty(rec.shape, dtype=newdtype).view(type(rec))

    for field in rec.dtype.fields:
        newrec[field] = rec[field]
        try:
            newrec.mask[field] = rec.mask[field]
        except AttributeError:
            pass #Not a masked array

    for n,a in zip(name, arr):
        newrec[n] = a
        try:
            newrec.mask[n] = np.array([False]*a.size).reshape(a.shape)
        except:
            pass
    return newrec
