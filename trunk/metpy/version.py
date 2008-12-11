import os.path
release = False
__version__ = '0.1'

_repository_path = os.path.split(__file__)[0]
_svn_file_path = os.path.join(_repository_path, '__svn_version__.py')

def get_svn_revision():
    '''
    Gets the "global" SVN revision number for the repository, using the
    path to this file.
    '''
    from subprocess import Popen, PIPE
    proc = Popen(['svnversion', '-n', _repository_path], stdout=PIPE)
    text = proc.stdout.readline()
    if text != 'exported':
        if ':' in text:
            text = text[text.find(':')+1:]
        rev = '.dev' + text
    else:
        rev = ''
    return rev

def write_svn_version():
    'Write the SVN revision to a file.'
    svnfile = open(_svn_file_path, 'w')
    svnfile.write('rev = "%s"\n' % get_svn_revision())
    svnfile.close()

def get_version():
    '''
    Get the version of the package, including the SVN revision if this
    is an actual release.
    '''
    version = __version__
    if not release:
        try:
            import __svn_version__
            version += __svn_version__.rev
        except ImportError:
            version += get_svn_revision()
            pass
    return version
