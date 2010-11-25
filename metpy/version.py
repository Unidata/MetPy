import os.path
import subprocess
release = False
__version__ = '0.1'


_repository_path = os.path.split(__file__)[0]
_git_file_path = os.path.join(_repository_path, '__git_version__.py')


def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
    return out

def get_git_hash():
    '''
    Gets the last GIT commit hash and date for the repository, using the
    path to this file.
    '''
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except:
        GIT_REVISION = None

    return GIT_REVISION

def get_git_revision():
    hash = get_git_hash()
    if hash :
        rev = '.dev.' + hash[:7]
        try:
            cmd = ['git', 'show', '%s' % (hash), '--date=short',
                   '--format="(%ad)"']
            date = _minimal_ext_cmd(cmd).split('"')[1]
            rev += date
        except:
            pass
    else:
        rev = ".dev.Unknown"

    return rev

def write_git_version():
    'Write the GIT revision to a file.'
    rev = get_git_revision()
    if rev == ".dev.Unknown":
        if os.path.isfile(_git_file_path):
            return
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
            version += __git_version__.rev
        except ImportError:
            version += get_git_revision()
    
    return version
