#!/usr/bin/env python
import os.path
import sys

with open(sys.argv[1], 'r') as infile:
    info = []
    for lineno,line in enumerate(infile):
        parts = line.split('  ')
        try:
            if len(parts) == 8:
                parts = parts[:6] + [parts[6] + parts[7]]

            var_name,desc,typ,units,rng,prec,byte_range = parts
            start,end = map(int, byte_range.split('-'))
            size = end - start + 1
            assert size >= 4
            if typ == 'Real*4':
                typ = 'f'
                assert size == 4, 'Got size %d' % size
            elif typ == 'Integer*4':
                typ = 'L'
                assert size == 4, 'Got size %d' % size
            elif typ == 'SInteger*4':
                typ = 'l'
                assert size == 4, 'Got size %d' % size
            elif typ.startswith('String'):
                typ = '%ds' % size
            elif not typ or typ == 'N/A':
                if var_name != 'SPARE':
                    print('WARNING: %s has type %s. Setting as SPARE' % (var_name, typ))
                    var_name = 'SPARE'
                typ = '%dx' % size
            elif typ == 'See Note (5)':
                typ = '%ds' % size
                assert size == 1172
            else:
                raise ValueError('No type match! (%s)' % typ)
            full_desc = desc.strip()
            if units and units != 'N/A':
                full_desc += ' (' + units + ')'

            name = var_name.strip()
            for char in '().':
                name = name.replace(char, '_')
            if name.endswith('_'):
                name = name[:-1]
            info.append((name, full_desc, typ))

            if ' ' in var_name:
                print('WARNING: space in %s' % var_name)
            if not desc:
                print('WARNING: null description for %s' % var_name)
        except ValueError:
            print('%d > %s' % (lineno + 1, ':'.join(parts)))
            raise

with open(os.path.splitext(sys.argv[1])[0] + '.py', 'w') as outfile:
    # File header
    outfile.write('# Generated file -- do not modify\n')

    # Variable descriptions
    outfile.write('descriptions = {')
    outdata = '\n    '.join('"{0}" : "{1}",'.format(*i) for i in info if i[0] != 'SPARE')
    outfile.write(outdata)
    outfile.write('}\n\n')

    # Now the struct format
    outfile.write('fields = [')
    outdata = '\n    '.join('({0}, "{1}"),'.format('"' + name + '"' if name != 'SPARE' else None, fmt)
            for name,__,fmt in info)
    outfile.write(outdata)
    outfile.write(']\n')
