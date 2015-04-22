import os.path
release = True
__version__ = '0.1.0'


_repository_path = os.path.split(__file__)[0]
_git_file_path = os.path.join(_repository_path, '__git_version__.py')


def get_revision():
    '''
    Gets the last GIT commit hash the repository, using the
    path to this file.
    '''
    package_dir = os.path.dirname(__file__)
    checkout_dir = os.path.normpath(os.path.join(package_dir, os.pardir))
    path = os.path.join(checkout_dir, '.git')
    if os.path.exists(path):
        return _get_git_revision(path)
    return None


def _get_git_revision(path):
    head = os.path.join(path, 'HEAD')
    if not os.path.exists(head):
        return None
    link = open(head, 'r').read().strip()
    if not link.startswith('ref:'):
        shorthash = link[:7]
    else:
        ref = link.split()[-1]
        shorthash = open(os.path.join(path, ref), 'r').read().strip()[:7]
    return shorthash


def write_git_version():
    'Write the GIT revision to a file.'
    rev = get_revision()
    gitfile = open(_git_file_path, 'w')
    gitfile.write('rev = "%s"\n' % rev)
    gitfile.close()


def get_version():
    '''
    Get the version of the package, including the GIT revision if this
    is an actual release.
    '''
    version = __version__
    if not release:
        try:
            import __git_version__
            rev = __git_version__.rev
        except ImportError:
            rev = get_revision()
        version += '.dev+' + rev

    return version
