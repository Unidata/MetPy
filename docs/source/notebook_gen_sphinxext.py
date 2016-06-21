#
# Generation of RST from notebooks
#
import glob
import os
import os.path
import warnings

warnings.simplefilter('ignore')

from nbconvert.exporters import rst


def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

    app.connect('builder-inited', generate_rst)

    return dict(
        version='0.1',
        parallel_read_safe=True,
        parallel_write_safe=True
    )

notebook_source_dir = '../../examples/notebooks'
generated_source_dir = 'examples/generated'


def nb_to_rst(nb_path):
    """convert notebook to restructured text"""
    exporter = rst.RSTExporter()
    out, resources = exporter.from_file(open(nb_path))
    basename = os.path.splitext(os.path.basename(nb_path))[0]
    imgdir = basename + '_files'
    img_prefix = os.path.join(imgdir, basename + '_')
    resources['metadata']['basename'] = basename
    resources['metadata']['name'] = basename.replace('_', ' ')
    resources['metadata']['imgdir'] = imgdir
    base_url = ('http://nbviewer.ipython.org/github/metpy/MetPy/blob/master/'
                'examples/notebooks/')
    out_lines = ['`Notebook <%s>`_' % (base_url + os.path.basename(nb_path))]
    for line in out.split('\n'):
        if line.startswith('.. image:: '):
            line = line.replace('output_', img_prefix)
        out_lines.append(line)
    out = '\n'.join(out_lines)

    return out, resources


def write_nb(dest, output, resources):
    if not os.path.exists(dest):
        os.makedirs(dest)
    rst_file = os.path.join(dest,
                            resources['metadata']['basename'] + resources['output_extension'])
    name = resources['metadata']['name']
    with open(rst_file, 'w') as rst:
        header = '=' * len(name)
        rst.write(header.encode('utf-8') + b'\n')
        rst.write(name.encode('utf-8') + b'\n')
        rst.write(header.encode('utf-8') + b'\n')
        rst.write(output.encode('utf-8'))

    imgdir = os.path.join(dest, resources['metadata']['imgdir'])
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)
    basename = resources['metadata']['basename']
    for filename in resources['outputs']:
        img_file =  os.path.join(imgdir, filename.replace('output_', basename + '_'))
        with open(img_file, 'wb') as img:
            img.write(resources['outputs'][filename])


def generate_rst(app):

    for fname in glob.glob(os.path.join(app.srcdir, notebook_source_dir, '*.ipynb')):
        write_nb(os.path.join(app.srcdir, generated_source_dir), *nb_to_rst(fname))
    with open(os.path.join(app.srcdir, 'examples', 'index.rst'), 'w') as test:
        test.write('==============\n''MetPy Examples\n''==============\n'
                   '.. toctree::\n   :glob:\n   :hidden:\n\n   generated/*\n\n')
        no_images = []
        for fname in glob.glob(os.path.join(app.srcdir, generated_source_dir, '*.rst')):
            filepath, filename = os.path.split(fname)
            target = filename.replace('.rst', '.html')
            dir = os.listdir(os.path.join(app.srcdir, generated_source_dir, filename.replace('.rst', '_files')))
            if dir:
                file = dir[0]
                test.write('.. image:: generated/' + filename.replace('.rst', '_files') + '/' + file +
                           '\n   :width: 220px'
                           '\n   :target: generated/' + target + '\n\n')
            else:
                no_images.append(target)
        for filename in no_images:
            test.write('`' + filename.replace('_', ' ').replace('.html', '') +
                       ' <generated/' + filename + '>`_\n\n')
