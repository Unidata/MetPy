'''Collection of generally useful utility code from the cookbook'''

from operator import itemgetter as _itemgetter
from keyword import iskeyword as _iskeyword
import itertools
import sys as _sys
import numpy as np
from numpy import ma

from matplotlib.cbook import iterable, is_string_like, Bunch

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
# These can be removed once numpy 1.3 is released
#
try:
    from numpy import mafromtxt
except ImportError:
    from genloadtxt import mloadtxt as mafromtxt

try:
    from numpy import ndfromtxt
except ImportError:
    from genloadtxt import loadtxt as ndfromtxt

try:
    from numpy.lib.recfunctions import stack_arrays
except ImportError:
    def stack_arrays(arrays, defaults=None, usemask=True, asrecarray=False,
                     autoconvert=False):
        """
        Superposes arrays fields by fields

        Parameters
        ----------
        seqarrays : array or sequence
            Sequence of input arrays.
        defaults : dictionary, optional
            Dictionary mapping field names to the corresponding default values.
        usemask : {True, False}, optional
            Whether to return a MaskedArray (or MaskedRecords is `asrecarray==True`)
            or a ndarray.
        asrecarray : {False, True}, optional
            Whether to return a recarray (or MaskedRecords if `usemask==True`) or
            just a flexible-type ndarray.
        autoconvert : {False, True}, optional
            Whether automatically cast the type of the field to the maximum.

        Examples
        --------
        >>> x = np.array([1, 2,])
        >>> stack_arrays(x) is x
        True
        >>> z = np.array([('A', 1), ('B', 2)], dtype=[('A', '|S3'), ('B', float)])
        >>> zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                          dtype=[('A', '|S3'), ('B', float), ('C', float)])
        >>> test = stack_arrays((z,zz))
        >>> masked_array(data = [('A', 1.0, --) ('B', 2.0, --) ('a', 10.0, 100.0)
        ... ('b', 20.0, 200.0) ('c', 30.0, 300.0)],
        ...       mask = [(False, False, True) (False, False, True) (False, False, False)
        ... (False, False, False) (False, False, False)],
        ...       fill_value=('N/A', 1e+20, 1e+20)
        ...       dtype=[('A', '|S3'), ('B', '<f8'), ('C', '<f8')])

        """
        if isinstance(arrays, ndarray):
            return arrays
        elif len(arrays) == 1:
            return arrays[0]
        seqarrays = [np.asanyarray(a).ravel() for a in arrays]
        nrecords = [len(a) for a in seqarrays]
        ndtype = [a.dtype for a in seqarrays]
        fldnames = [d.names for d in ndtype]
        #
        dtype_l = ndtype[0]
        newdescr = dtype_l.descr
        names = [_[0] for _ in newdescr]
        for dtype_n in ndtype[1:]:
            for descr in dtype_n.descr:
                name = descr[0] or ''
                if name not in names:
                    newdescr.append(descr)
                    names.append(name)
                else:
                    nameidx = names.index(name)
                    current_descr = newdescr[nameidx]
                    if autoconvert:
                        if np.dtype(descr[1]) > np.dtype(current_descr[-1]):
                            current_descr = list(current_descr)
                            current_descr[-1] = descr[1]
                            newdescr[nameidx] = tuple(current_descr)
                    elif descr[1] != current_descr[-1]:
                        raise TypeError("Incompatible type '%s' <> '%s'" %\
                                        (dict(newdescr)[name], descr[1]))
        # Only one field: use concatenate
        if len(newdescr) == 1:
            output = ma.concatenate(seqarrays)
        else:
            #
            output = ma.masked_all((np.sum(nrecords),), newdescr)
            offset = np.cumsum(np.r_[0, nrecords])
            seen = []
            for (a, n, i, j) in zip(seqarrays, fldnames, offset[:-1], offset[1:]):
                names = a.dtype.names
                if names is None:
                    output['f%i' % len(seen)][i:j] = a
                else:
                    for name in n:
                        output[name][i:j] = a[name]
                        if name not in seen:
                            seen.append(name)
        #
        return _fix_output(_fix_defaults(output, defaults),
                           usemask=usemask, asrecarray=asrecarray)

try:
    from numpy.lib.recfunctions import append_fields
except ImportError:
#Taken from a numpy-discussion mailing list post 'Re: adding field to rec array'
#by Robert Kern.  Modified to handle masked arrays, which is why we don't
#just use the matplotlib version

    def append_fields(rec, names, arr, dtype=None):
        """
        Appends a field to an existing record array, handling masked fields
        if necessary.

        Parameters
        ----------
        rec : numpy record array
            Array to which the new field should be appended
        names : string
            Names to be given to the new fields
        arr : ndarray
            Array containing the data for the new fields.
        dtype : data-type or None, optional
            Data type of the new fields.  If this is None, the data types will
            be obtained from `arr`.

        Returns
        -------
        out : numpy record array
            `rec` with the new field appended.
        rec = append_fields(rec, name, arr)
        """
        if not iterable(names):
            names = [names]
        if not iterable(arr):
            arr = [arr]

        if dtype is None:
            dtype = [a.dtype for a in arr]

        newdtype = np.dtype(rec.dtype.descr + zip(names, dtype))
        newrec = np.empty(rec.shape, dtype=newdtype).view(type(rec))

        for name in rec.dtype.names:
            newrec[name] = rec[name]
            try:
                newrec.mask[name] = rec.mask[name]
            except AttributeError:
                pass #Not a masked array

        for n,a in zip(names, arr):
            newrec[n] = a
            try:
                old_mask = a.mask
            except AttributeError:
                old_mask = np.array([False]*a.size).reshape(a.shape)
            try:
                newrec[n].mask = old_mask
            except AttributeError:
                pass
        return newrec

def add_dtype_titles(array, title_map):
    '''
    Add titles to the fields in the array, handling masked arrays if
    necessary.

    array : ndarray
        The array to which to add the titles

    title_map : dictionary
        A dictionary mapping field names to the titles that should be
        added to the field.

    Returns : None
    '''
    # Loop over all fields in the dtype and add a title to the tuple in
    # the field.  Make a new dtype from this list
    newdtype = np.dtype(dict((name, info + (title_map.get(name, None),))
            for name,info in array.dtype.fields.iteritems()))
    new_array = np.empty(array.shape, dtype=newdtype).view(type(array))

    for name in new_array.dtype.names:
        new_array[name] = array[name]
        try:
            new_array[name].mask = array[name].mask
        except AttributeError:
            pass

    return new_array

def get_title(array, name):
    '''
    Fetch the title for field *name*.

    array : ndarray
        The data type

    name : string
        The field name

    Returns : object
        The object set as the field's title. If there is none, just
        return the name.
    '''
    try:
        field = array.dtype.fields[name]
    except (TypeError, KeyError, AttributeError):
        return name
    else:
        if len(field) > 2:
            return field[-1]
        return name
