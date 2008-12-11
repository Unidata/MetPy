release = False
__version__ = '0.1'

def get_svn_version():
    import subprocess
    proc = subprocess.Popen(['svnversion', '-n'], stdout=subprocess.PIPE)
    text = proc.stdout.readline()
    if text != 'exported':
        if ':' in text:
            text = text[text.find(':')+1:]
        rev = '.dev' + text
    else:
        rev = ''
    return rev

def write_svn_version():
    import os.path

    rev = get_svn_version()
    libpath = os.path.split(os.path.abspath(__file__))[0]
    svnfile = open(os.path.join(libpath, '__svn_version__.py'), 'w')
    svnfile.write('rev = "%s"\n' % rev)
    svnfile.close()

def get_version():
    version = __version__
    if not release:
        try:
            import __svn_version__
            version += __svn_version__.rev
        except ImportError:
            print 'local run'
            version += get_svn_version()
            pass
    return version
