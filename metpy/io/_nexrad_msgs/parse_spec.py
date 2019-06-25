#!/usr/bin/env python
# Copyright (c) 2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Parse specification extracted from NEXRAD ICD PDFs and generate Python code."""

from __future__ import print_function

import warnings


def register_processor(num):
    """Register functions to handle particular message numbers."""
    def inner(func):
        """Perform actual function registration."""
        processors[num] = func
        return func
    return inner


processors = {}


@register_processor(3)
def process_msg3(fname):
    """Handle information for message type 3."""
    with open(fname, 'r') as infile:
        info = []
        for lineno, line in enumerate(infile):
            parts = line.split('  ')
            try:
                var_name, desc, typ, units = parts[:4]
                size_hw = parts[-1]
                if '-' in size_hw:
                    start, end = map(int, size_hw.split('-'))
                    size = (end - start + 1) * 2
                else:
                    size = 2

                assert size >= 2
                fmt = fix_type(typ, size)

                var_name = fix_var_name(var_name)
                full_desc = fix_desc(desc, units)

                info.append({'name': var_name, 'desc': full_desc, 'fmt': fmt})

                if ignored_item(info[-1]) and var_name != 'Spare':
                    warnings.warn('{} has type {}. Setting as Spare'.format(var_name, typ))

            except (ValueError, AssertionError):
                warnings.warn('{} > {}'.format(lineno + 1, ':'.join(parts)))
                raise
        return info


@register_processor(18)
def process_msg18(fname):
    """Handle information for message type 18."""
    with open(fname, 'r') as infile:
        info = []
        for lineno, line in enumerate(infile):
            parts = line.split('  ')
            try:
                if len(parts) == 8:
                    parts = parts[:6] + [parts[6] + parts[7]]

                var_name, desc, typ, units, rng, prec, byte_range = parts
                start, end = map(int, byte_range.split('-'))
                size = end - start + 1
                assert size >= 4
                fmt = fix_type(typ, size,
                               additional=[('See Note (5)', ('{size}s', 1172))])

                if ' ' in var_name:
                    warnings.warn('Space in {}'.format(var_name))
                if not desc:
                    warnings.warn('null description for {}'.format(var_name))

                var_name = fix_var_name(var_name)
                full_desc = fix_desc(desc, units)

                info.append({'name': var_name, 'desc': full_desc, 'fmt': fmt})

                if (ignored_item(info[-1]) and var_name != 'SPARE'
                        and 'SPARE' not in full_desc):
                    warnings.warn('{} has type {}. Setting as SPARE'.format(var_name, typ))

            except (ValueError, AssertionError):
                warnings.warn('{} > {}'.format(lineno + 1, ':'.join(parts)))
                raise
        return info


types = [('Real*4', ('f', 4)), ('Integer*4', ('L', 4)), ('SInteger*4', ('l', 4)),
         ('Integer*2', ('H', 2)),
         ('', lambda s: ('{size}x', s)), ('N/A', lambda s: ('{size}x', s)),
         (lambda t: t.startswith('String'), lambda s: ('{size}s', s))]


def fix_type(typ, size, additional=None):
    """Fix up creating the appropriate struct type based on the information in the column."""
    if additional is not None:
        my_types = types + additional
    else:
        my_types = types

    for t, info in my_types:
        if callable(t):
            matches = t(typ)
        else:
            matches = t == typ

        if matches:
            if callable(info):
                fmt_str, true_size = info(size)
            else:
                fmt_str, true_size = info
            assert size == true_size, ('{}: Got size {} instead of {}'.format(typ, size,
                                                                              true_size))
            return fmt_str.format(size=size)

    raise ValueError('No type match! ({})'.format(typ))


def fix_var_name(var_name):
    """Clean up and apply standard formatting to variable names."""
    name = var_name.strip()
    for char in '(). /#,':
        name = name.replace(char, '_')
    name = name.replace('+', 'pos_')
    name = name.replace('-', 'neg_')
    if name.endswith('_'):
        name = name[:-1]
    return name


def fix_desc(desc, units=None):
    """Clean up description column."""
    full_desc = desc.strip()
    if units and units != 'N/A':
        if full_desc:
            full_desc += ' (' + units + ')'
        else:
            full_desc = units
    return full_desc


def ignored_item(item):
    """Determine whether this item should be ignored."""
    return item['name'].upper() == 'SPARE' or 'x' in item['fmt']


def need_desc(item):
    """Determine whether we need a description for this item."""
    return item['desc'] and not ignored_item(item)


def field_name(item):
    """Return the field name if appropriate."""
    return '"{:s}"'.format(item['name']) if not ignored_item(item) else None


def field_fmt(item):
    """Return the field format if appropriate."""
    return '"{:s}"'.format(item['fmt']) if '"' not in item['fmt'] else item['fmt']


def write_file(fname, info):
    """Write out the generated Python code."""
    with open(fname, 'w') as outfile:
        # File header
        outfile.write('# Copyright (c) 2018 MetPy Developers.\n')
        outfile.write('# Distributed under the terms of the BSD 3-Clause License.\n')
        outfile.write('# SPDX-License-Identifier: BSD-3-Clause\n\n')
        outfile.write('# flake8: noqa\n')
        outfile.write('# Generated file -- do not modify\n')

        # Variable descriptions
        outfile.write('descriptions = {')
        outdata = ',\n                '.join('"{name}": "{desc}"'.format(
            **i) for i in info if need_desc(i))
        outfile.write(outdata)
        outfile.write('}\n\n')

        # Now the struct format
        outfile.write('fields = [')
        outdata = ',\n          '.join('({fname}, "{fmt}")'.format(
            fname=field_name(i), **i) for i in info)
        outfile.write(outdata)
        outfile.write(']\n')


if __name__ == '__main__':
    import os.path

    for num in [18, 3]:
        fname = 'msg{:d}.spec'.format(num)
        print('Processing {}...'.format(fname))  # noqa: T001
        info = processors[num](fname)
        fname = os.path.splitext(fname)[0] + '.py'
        write_file(fname, info)
