#!/usr/bin/env python
# This script is used to convert colortables from GRLevelX to data for a
# matplotlib colormap
import sys
from optparse import OptionParser

#Set up command line options
opt_parser = OptionParser(usage="usage: %prog [options] colortablefile")
opt_parser.add_option("-s", "--scale", action="store_true", dest="scale",
    help="Scale size of colortable entries by thresholds in file.")

opts,args = opt_parser.parse_args()
if not args:
   print "You must pass the colortable file as the commandline argument."
   opt_parser.print_help()
   sys.exit(-1)

fname = args[0]
scaleTable = opts.scale

colors = []
thresholds = []
#Initial color should end up not used by MPL
prev = [0., 0., 0.]
for line in open(fname, 'r'):
    if line.startswith("Color:"):
        # This ignores the word "Color:" and the threshold
        # and converts the rest to float
        parts = line.split()
        thresholds.append(float(parts[1]))

        color_info = [float(x)/255. for x in parts[2:]]
        if not prev:
            prev = info[:3]

        colors.append(zip(prev, color_info[:3]))

        prev = color_info[3:]

# Add the last half of the last line, if necessary
if prev:
    colors.append(zip(prev,prev))

colordict = dict(red=[], green=[], blue=[])
num_entries = float(len(colors) - 1)
offset = min(thresholds)
scale = 1. / (max(thresholds) - offset)
for i,(t,(r,g,b)) in enumerate(zip(thresholds, colors)):
    if scaleTable:
        norm = (t - offset) * scale
    else:
        norm = i / num_entries

    colordict['red'].append((norm,) + r)
    colordict['green'].append((norm,) + g)
    colordict['blue'].append((norm,) + b)

# Output as code that can be copied and pasted into a python script. Repr()
# would work here, but wouldn't be as human-readable.
print '{'
num_colors = len(colordict.keys())
for n,k in enumerate(sorted(colordict.keys())):
    print "'%s' :" % k
    num = len(colordict[k])
    for i,line in enumerate(colordict[k]):
        if i == 0:
            print '    [%s,' % repr(line)
        elif i == num - 1:
            if n == num_colors - 1:
                print '    %s]' % repr(line)
            else:
                print '    %s],' % repr(line)
        else:
            print "    %s," % repr(line)
print '}'

if not scaleTable:
    print repr(thresholds)
