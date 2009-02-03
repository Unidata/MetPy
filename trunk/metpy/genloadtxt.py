"""
Proposal :
Here's an extension to np.loadtxt, designed to take missing values into account.

"""



import itertools
import numpy as np
import numpy.ma as ma


def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True

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
    if _is_string_like(fname):
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
    delimiter : var, optional
        If a string, character used to delimit consecutive fields.
        If an integer or a sequence of integers, width(s) of each field.
    autostrip : boolean, optional
        Whether to strip each individual fields
    """

    def autostrip(self, method):
        "Wrapper to strip each member of the output of `method`."
        return lambda input: [_.strip() for _ in method(input)]
    #
    def __init__(self, delimiter=None, comments='#', autostrip=True):
        self.comments = comments
        # Delimiter is a character
        if (delimiter is None) or _is_string_like(delimiter):
            delimiter = delimiter or None
            _handyman = self._delimited_splitter
        # Delimiter is a list of field widths
        elif hasattr(delimiter, '__iter__'):
            _handyman = self._variablewidth_splitter
            idx = np.cumsum([0]+list(delimiter))
            delimiter = [slice(i,j) for (i,j) in zip(idx[:-1], idx[1:])]
        # Delimiter is a single integer
        elif int(delimiter):
            (_handyman, delimiter) = (self._fixedwidth_splitter, int(delimiter))
        else:
            (_handyman, delimiter) = (self._delimited_splitter, None)
        self.delimiter = delimiter
        if autostrip:
            self._handyman = self.autostrip(_handyman)
        else:
            self._handyman = _handyman
    #
    def _delimited_splitter(self, line):
        line = line.split(self.comments)[0].strip()
        if not line:
            return []
        return line.split(self.delimiter)
    #
    def _fixedwidth_splitter(self, line):
        line = line.split(self.comments)[0]
        if not line:
            return []
        fixed = self.delimiter
        slices = [slice(i, i+fixed) for i in range(len(line))[::fixed]]
        return [line[s] for s in slices]
    #
    def _variablewidth_splitter(self, line):
        line = line.split(self.comments)[0]
        if not line:
            return []
        slices = self.delimiter
        return [line[s] for s in slices]
    #
    def __call__(self, line):
        return self._handyman(line)



class NameValidator:
    """
    Validates a list of strings to use as field names.
    The strings are stripped of any non alphanumeric character, and spaces
    are replaced by `_`. If the optional input parameter `case_sensitive`
    is False, the strings are set to upper case.

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
    casesensitive : boolean, optional
        Whether the field names are case sensitive or not. If not, then fields
        like `'date'` and `'DATE'` are assumed to be the same.
    """
    #
    defaultexcludelist = ['return','file','print']
    defaultdeletechars = set("""~!@#$%^&*()-=+~\|]}[{';: /?.>,<""")
    #
    def __init__(self, excludelist=None, deletechars=None, case_sensitive=True):
        #
        if excludelist is None:
            excludelist = []
        excludelist.extend(self.defaultexcludelist)
        self.excludelist = excludelist
        #
        if deletechars is None:
            delete = self.defaultdeletechars
        else:
            delete = set(deletechars)
        delete.add('"')
        self.deletechars = delete
        self.case_sensitive = case_sensitive
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
        casesensitive = self.case_sensitive
        for i, item in enumerate(names):
            if not casesensitive:
                item = item.upper()
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
    type : type
        Type of the output.
    _status : integer
        Integer representing the order of the conversion.
    _mapper : sequence of tuples
        Sequence of tuples (dtype, function, default value) to evaluate in order.
    _locked : boolean
        Whether the StringConverter is locked, thereby preventing automatic any
        upgrade or not.

    """
    #
    _mapper = [(np.bool_, str2bool, False),
               (np.integer, int, -1),
               (np.floating, float, np.nan),
               (complex, complex, np.nan+0j),
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

    Examples
    --------
    >>> import dateutil.parser
    >>> import datetime
    >>> dateparser = datetutil.parser.parse
    >>> defaultdate = datetime.date(2000, 1, 1)
    >>> StringConverter.upgrade_mapper(dateparser, default=defaultdate)
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
            self.default = default or False
            ttype = np.bool
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
        self._callingfunction = self._strict_call
        self.type = ttype
        self._checked = False
    #
    def _loose_call(self, value):
        try:
            return self.func(value)
        except ValueError:
            return self.default
    #
    def _strict_call(self, value):
        try:
            return self.func(value)
        except ValueError:
            if value.strip() in self.missing_values:
                if not self._status:
                    self._checked = False
                return self.default
            raise ValueError("Cannot convert string '%s'" % value)
    #
    def __call__(self, value):
        return self._callingfunction(value)
    #
    def upgrade(self, value):
        """
    Tries to find the best converter for `value`, by testing different
    converters in order.
    The order in which the converters are tested is read from the
    :attr:`_status` attribute of the instance.
        """
        self._checked = True
        try:
            self._strict_call(value)
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
            (self.type, self.func, self.default) = self._mapper[self._status]
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
            if _is_string_like(missing_values):
                self.missing_values.add(missing_values)
            elif hasattr(missing_values, '__iter__'):
                for val in missing_values:
                    self.missing_values.add(val)
        else:
            self.missing_values = []
        # Update the type
        try:
            tester = func('0')
        except ValueError:
            tester = None
        self.type = self._getsubdtype(tester)




def genloadtxt(fname, dtype=float, comments='#', delimiter=None, skiprows=0,
               converters=None, missing='', missing_values=None, usecols=None,
               names=None, excludelist=None, deletechars=None,
               unpack=None, usemask=False, loose=True):
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
        The string used to separate values.  By default, any consecutive
        whitespace act as delimiter.
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
    usemask : {bool}, optional
        Whether to create a mask indicating where data is missing.
    loose : {bool}, optional
        Whether to use a loose converter or not. With a loose converter,
        data that cannot be converted is transformed to a default value,
        and no ValueError exception is raised.


    Returns
    -------
    out : MaskedArray
        Data read from the text file.

    Notes
    --------
    * When spaces are used as delimiters, or when no delimiter has been given
      as input, there should not be any missing data between two fields.
    * When `names` is not None, names are lower cased, the spaces replaced by
      underscores, and any illegal character suppressed.
    * When the variable are named (either by a flexible dtype or with `names`,
      there must not be any header in the file (else a :exc:ValueError exception
      is raised).


    """
    # Check the input dictionary of converters
    user_converters = converters or {}
    if not isinstance(user_converters, dict):
        errmsg = "The input argument 'converter' should be a valid dictionary "\
                 "(got '%s' instead)"
        raise TypeError(errmsg % type(user_converters))
    # Check the input dictionary of missing values
    user_missing_values = missing_values or {}
    if not isinstance(user_missing_values, dict):
        errmsg = "The input argument 'missing_values' should be a valid "\
                 "dictionary (got '%s' instead)"
        raise TypeError(errmsg % type(missing_values))
    defmissing = [_.strip() for _ in missing.split(',')] + ['']

    # Initialize the filehandle, the LineSplitter and the NameValidator
    fhd = _to_filehandle(fname)
    split_line = LineSplitter(delimiter=delimiter, comments=comments,
                              autostrip=False)._handyman
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

    # Check the names and overwrite the dtype.names if needed
    if dtype is not None:
        dtype = np.dtype(dtype)
    dtypenames = getattr(dtype, 'names', None)
    if names is True:
        names = validate_names([_.strip() for _ in first_values])
        first_line =''
    elif _is_string_like(names):
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
            if _is_string_like(current):
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
        if _is_string_like(key):
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
                              missing_values=missing_values.get(i, defmissing),
                              locked=True)
                          for (i, dt) in enumerate(flatdtypes)]
        else:
            # Set to a default converter (but w/ different missing values)
            converters = [StringConverter(dtype,
                              missing_values=missing_values.get(_, defmissing),
                              locked=True)
                          for _ in range(nbcols)]
    missing_values = [_.missing_values for _ in converters]

    # Update the converters to use the user-defined ones
    for (i, conv) in user_converters.iteritems():
        # If the converter is specified by column names, use the index instead
        if _is_string_like(i):
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
    all_locked = min((_._locked for _ in converters))

    # Reset the names to match the usecols
    if (not first_line) and usecols:
        names = [names[_] for _ in usecols]

    rows = []
    append_to_rows = rows.append
    if usemask:
        masks = []
        append_to_masks = masks.append
    # Parse each line
    for line in itertools.chain([first_line,], fhd):
        values = split_line(line)
        # Skip an empty line
        if len(values) == 0:
            continue
        # Select only the columns we need
        if usecols:
            values = [values[_] for _ in usecols]
        # Check whether we need to update the converter
        if dtype is None:
            for (converter, item) in zip(converters, values):
                converter.upgrade(item)
        # Store the values
        append_to_rows(tuple(values))
        if usemask:
            append_to_masks(tuple([val.strip() in mss
                                   for (val, mss) in zip(values,
                                                         missing_values)]))

    # Convert each value according to the converter:
    # We want to modify the list in place to avoid creating a new one...
    if loose:
        conversionfuncs = [conv._loose_call for conv in converters]
    else:
        conversionfuncs = [conv._strict_call for conv in converters]
    for (i, vals) in enumerate(rows):
        rows[i] = tuple([convert(val)
                         for (convert, val) in zip(conversionfuncs, vals)])


    # Reset the dtype
    data = rows
    if dtype is None:
        # Get the dtypes from the types of the converters
        coldtypes = [conv.type for conv in converters]
        # Find the columns with strings...
        strcolidx = [i for (i, v) in enumerate(coldtypes)
                     if v in (type('S'), np.string_)]
        # ... and take the largest number of chars.
        for i in strcolidx:
            coldtypes[i] = "|S%i" % max(len(row[i]) for row in data)
        #
        if names is None:
            # If the dtype is uniform, don't define names, else use ''
            base = set([c.type for c in converters if c._checked])

            if len(base) == 1:
                (ddtype, mdtype) = (list(base)[0], np.bool)
            else:
                ddtype = [('', dt) for dt in coldtypes]
                mdtype = [('', np.bool) for dt in coldtypes]
        else:
            ddtype = zip(names, coldtypes)
            mdtype = zip(names, [np.bool] * len(coldtypes))
        output = np.array(data, dtype=ddtype)
        if usemask:
            outputmask = np.array(masks, dtype=mdtype)
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
            rows = np.array(data, dtype=[('', t) for t in flatdtypes])
            output = rows.view(dtype)
            # Now, process the rowmasks the same way
            if usemask:
                rowmasks = np.array(masks,
                                    dtype=np.dtype([('', np.bool)
                                                    for t in flatdtypes]))
                # Construct the new dtype
                mdtype = nested_masktype(dtype)
                outputmask = rowmasks.view([tuple(_) for _ in mdtype])
        else:
            output = np.array(data, dtype)
            if usemask:
                if dtype.names:
                    mdtype = [(_, np.bool) for _ in dtype.names]
                else:
                    mdtype = np.bool
                outputmask = np.array(masks, dtype=mdtype)
    # Construct the final array
    if unpack:
        if usemask:
            return (output.squeeze().T, outputmask.squeeze().T)
        return (output.squeeze().T, None)
    if usemask:
        return (output.squeeze(), outputmask.squeeze().T)
    return (output.squeeze(), None)



def loadtxt(fname, dtype=float, comments='#', delimiter=None, skiprows=0,
               converters=None, missing='', missing_values=None,
               usecols=None, unpack=None,
               names=None, excludelist=None, deletechars=None):
    kwargs = dict(dtype=dtype, comments=comments, delimiter=delimiter,
                  skiprows=skiprows, converters=converters,
                  missing=missing, missing_values=missing_values,
                  usecols=usecols, unpack=unpack, names=names,
                  excludelist=excludelist, deletechars=deletechars,
                  usemask=False)
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
                  excludelist=excludelist, deletechars=deletechars,
                  usemask=True)
    (output, outputmask) = genloadtxt(fname, **kwargs)
    output = output.view(ma.MaskedArray)
    output.mask = outputmask
    return output

################################################################################
from numpy.ma.testutils import *
import StringIO
#from numpy import loadtxt


if __name__ == '__main__':
    import hotshot, hotshot.stats
    import os
    import tempfile
    (tmp_fd,tmp_fl) = tempfile.mkstemp()
    #
    # Create a fake dataset
    length = 5000
    data = np.empty((length,3), dtype="|S6")
    data.flat = np.array(np.random.rand(length*3))
#        data[np.random.randint(0,length*3-1,0.1*length)] = ''
    dfile = StringIO.StringIO()
    for row in data:
        os.write(tmp_fd, ", ".join(row) + "\n")
    os.close(tmp_fd)

    if 1:
        # Get a hotshot profile for genloadtxt
        kwargs = dict(delimiter=",", dtype=float)
        profiler = hotshot.Profile( "_preview.prof", lineevents=0 )
        output = profiler.runcall(genloadtxt, tmp_fl, **kwargs)
        profiler.close()
        stats = hotshot.stats.load("_preview.prof")
        stats.strip_dirs()
        stats.sort_stats('time', 'calls')
        stats.print_stats(20)

        # Get a hotshot profile for np.loadtxt
        kwargs = dict(delimiter=",", dtype=float)
        dfile.seek(0)
        profiler = hotshot.Profile( "_preview.prof", lineevents=0 )
        output = profiler.runcall(np.loadtxt, tmp_fl, **kwargs)
        profiler.close()
        stats = hotshot.stats.load("_preview.prof")
        stats.strip_dirs()
        stats.sort_stats('time', 'calls')
        stats.print_stats(20)
        #
        import timeit
        repeatargs = (3, 5)
        setup="""
from __main__ import loadtxt, mloadtxt
from matplotlib.mlab import csv2rec
import numpy as np
        """
        args = "'%s', dtype=float, delimiter=','" % tmp_fl
        #
        command = "np.loadtxt(%s)" % args
        timer = timeit.Timer(command, setup)
        timer_np = min(timer.repeat(*repeatargs))
        print command, timer_np
        output_np = np.loadtxt(tmp_fl, dtype=float, delimiter=',')
        #
        command = "loadtxt(%s)" % args
        timer = timeit.Timer(command, setup)
        timer_nw = min(timer.repeat(*repeatargs))
        print command, timer_nw, "(%+02.2f%%)" % ((timer_nw/timer_np-1)*100)
        output_nw = loadtxt(tmp_fl, dtype=float, delimiter=',')


        args = "'%s', delimiter=','" % tmp_fl
        command = "csv2rec(%s)" % args
        timer = timeit.Timer(command, setup)
        print command, min(timer.repeat(*repeatargs))
        from matplotlib.mlab import csv2rec
        output_ml = csv2rec(tmp_fl, delimiter=',', names=("a","b","c")).view((float,3))
    #    except:
    #        raise
    #    finally:
        os.remove(tmp_fl)
