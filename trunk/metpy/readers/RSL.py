from ctypes import *
import numpy as N

librslPath = '/usr/local/trmm/GVBOX/lib/librsl.dylib'


_libraries = {}
_libraries[librslPath] = CDLL(librslPath)
STRING = c_char_p


class fieldTypes:
	def __init__(self):
		self.DZ =  0
		self.VR =  1
		self.SW =  2
		self.CZ =  3
		self.ZT =  4
		self.DR =  5
		self.LR =  6
		self.ZD =  7
		self.DM =  8
		self.RH =  9
		self.PH = 10
		self.XZ = 11
		self.CD = 12
		self.MZ = 13
		self.MD = 14
		self.ZE = 15
		self.VE = 16
		self.KD = 17
		self.TI = 18
		self.DX = 19
		self.CH = 20
		self.AH = 21
		self.CV = 22
		self.AV = 23
		self.SQ = 24
		
		self.list = ['DZ',
            		 'VR',
            		 'SW',
            		 'CZ',
            		 'ZT',
            		 'DR',
            		 'LR',
            		 'ZD',
            		 'DM',
            		 'RH',
            		 'PH',
            		 'XZ',
            		 'CD',
            		 'MZ',
            		 'MD',
            		 'ZE',
            		 'VE',
            		 'KD',
            		 'TI',
            		 'DX',
            		 'CH',
            		 'AH',
            		 'CV',
            		 'AV',
            		 'SQ',]
                    

R_LR = 20
TOGA_FILE = 4
# def __IDSTRING(name,string): return static const char name[] __unused = string # macro
V_DR = 5
# def __sgetc(p): return (--(p)->_r < 0 ? __srget(p) : (int)(*(p)->_p++)) # macro
NSIG_FILE_V2 = 6
S_DZ = 7
def getchar_unlocked(): return getc_unlocked(stdin) # macro
# def FD_ISSET(n,p): return ((p)->fds_bits[(n)/NFDBITS] & (1 << ((n) % NFDBITS))) # macro
S_SW = 9
def htonl(x): return (x) # macro
RAPIC_FILE = 10
# def __offsetof(type,field): return ((size_t)(&((type *)0)->field)) # macro
S_ZT = 11
EDGE_FILE = 12
# def __sfeof(p): return (((p)->_flags & __SEOF) != 0) # macro
S_LR = 13
def minor(x): return ((int32_t)((x) & 0xffffff)) # macro
def FD_ZERO(p): return bzero(p, sizeof(*(p))) # macro
RAINBOW_FILE = 14
# def __sfileno(p): return ((p)->_file) # macro
R_VR = 15
HDF_FILE = 9
def ntohs(x): return (x) # macro
MCGILL_FILE = 8
def HTONS(x): return (x) # macro
def putchar_unlocked(x): return putc_unlocked(x, stdout) # macro
V_DZ = 0
def clearerr_unlocked(p): return __sclearerr(p) # macro
R_CZ = 17
WSR88D_FILE = 1
V_SW = 2
def __P(protos): return protos # macro
def ferror_unlocked(p): return __sferror(p) # macro
def major(x): return ((int32_t)(((u_int32_t)(x) >> 24) & 0xff)) # macro
LASSEN_FILE = 3
def __COPYRIGHT(s): return __IDSTRING(copyright,s) # macro
V_ZT = 4
NSIG_FILE_V1 = 5
V_LR = 6
def getc_unlocked(fp): return __sgetc(fp) # macro
RSL_FILE = 7
S_VR = 8
def howmany(x,y): return (((x) + ((y) - 1)) / (y)) # macro
def __STRING(x): return #x # macro
S_CZ = 10
def htons(x): return (x) # macro
# def __sclearerr(p): return ((void)((p)->_flags &= ~(__SERR|__SEOF))) # macro
RADTEC_FILE = 11
# def FD_SET(n,p): return ((p)->fds_bits[(n)/NFDBITS] |= (1 << ((n) % NFDBITS))) # macro
DORADE_FILE = 13
def __RCSID(s): return __IDSTRING(rcsid,s) # macro
# def __sferror(p): return (((p)->_flags & __SERR) != 0) # macro
R_DZ = 14
def ntohl(x): return (x) # macro
def HTONL(x): return (x) # macro
# def FD_CLR(n,p): return ((p)->fds_bits[(n)/NFDBITS] &= ~(1 << ((n) % NFDBITS))) # macro
S_DR = 12
R_SW = 16
def feof_unlocked(p): return __sfeof(p) # macro
def NTOHS(x): return (x) # macro
def putc_unlocked(x,fp): return __sputc(x, fp) # macro
UNKNOWN = 0
def NTOHL(x): return (x) # macro
V_VR = 1
R_DR = 19
UF_FILE = 2
R_ZT = 18
def __CONCAT(x,y): return x ## y # macro
V_CZ = 3
htonl = _libraries[librslPath].htonl
htonl.restype = c_ulong
htonl.argtypes = [c_ulong]
htons = _libraries[librslPath].htons
htons.restype = c_ushort
htons.argtypes = [c_ushort]
ntohl = _libraries[librslPath].ntohl
ntohl.restype = c_ulong
ntohl.argtypes = [c_ulong]
ntohs = _libraries[librslPath].ntohs
ntohs.restype = c_ushort
ntohs.argtypes = [c_ushort]
int8_t = c_byte
u_int8_t = c_ubyte
int16_t = c_short
u_int16_t = c_ushort
int32_t = c_int
u_int32_t = c_uint
int64_t = c_longlong
u_int64_t = c_ulonglong
register_t = int32_t
intptr_t = c_long
uintptr_t = c_ulong
quad_t = int64_t
off_t = quad_t
fpos_t = off_t
class __sbuf(Structure):
    pass
__sbuf._fields_ = [
    ('_base', POINTER(c_ubyte)),
    ('_size', c_int),
]
class __sFILEX(Structure):
    pass
__sFILEX._fields_ = [
]
class __sFILE(Structure):
    pass
__sFILE._fields_ = [
    ('_p', POINTER(c_ubyte)),
    ('_r', c_int),
    ('_w', c_int),
    ('_flags', c_short),
    ('_file', c_short),
    ('_bf', __sbuf),
    ('_lbfsize', c_int),
    ('_cookie', c_void_p),
    ('_close', CFUNCTYPE(c_int, c_void_p)),
    ('_read', CFUNCTYPE(c_int, c_void_p, STRING, c_int)),
    ('_seek', CFUNCTYPE(fpos_t, c_void_p, c_longlong, c_int)),
    ('_write', CFUNCTYPE(c_int, c_void_p, STRING, c_int)),
    ('_ub', __sbuf),
    ('_extra', POINTER(__sFILEX)),
    ('_ur', c_int),
    ('_ubuf', c_ubyte * 3),
    ('_nbuf', c_ubyte * 1),
    ('_lb', __sbuf),
    ('_blksize', c_int),
    ('_offset', fpos_t),
]
FILE = __sFILE
clearerr = _libraries[librslPath].clearerr
clearerr.restype = None
clearerr.argtypes = [POINTER(FILE)]
fclose = _libraries[librslPath].fclose
fclose.restype = c_int
fclose.argtypes = [POINTER(FILE)]
feof = _libraries[librslPath].feof
feof.restype = c_int
feof.argtypes = [POINTER(FILE)]
ferror = _libraries[librslPath].ferror
ferror.restype = c_int
ferror.argtypes = [POINTER(FILE)]
fflush = _libraries[librslPath].fflush
fflush.restype = c_int
fflush.argtypes = [POINTER(FILE)]
fgetc = _libraries[librslPath].fgetc
fgetc.restype = c_int
fgetc.argtypes = [POINTER(FILE)]
fgetpos = _libraries[librslPath].fgetpos
fgetpos.restype = c_int
fgetpos.argtypes = [POINTER(FILE), POINTER(fpos_t)]
fgets = _libraries[librslPath].fgets
fgets.restype = STRING
fgets.argtypes = [STRING, c_int, POINTER(FILE)]
fopen = _libraries[librslPath].fopen
fopen.restype = POINTER(FILE)
fopen.argtypes = [STRING, STRING]
fprintf = _libraries[librslPath].fprintf
fprintf.restype = c_int
fprintf.argtypes = [POINTER(FILE), STRING]
fputc = _libraries[librslPath].fputc
fputc.restype = c_int
fputc.argtypes = [c_int, POINTER(FILE)]
fputs = _libraries[librslPath].fputs
fputs.restype = c_int
fputs.argtypes = [STRING, POINTER(FILE)]
size_t = c_ulong
fread = _libraries[librslPath].fread
fread.restype = size_t
fread.argtypes = [c_void_p, size_t, size_t, POINTER(FILE)]
freopen = _libraries[librslPath].freopen
freopen.restype = POINTER(FILE)
freopen.argtypes = [STRING, STRING, POINTER(FILE)]
fscanf = _libraries[librslPath].fscanf
fscanf.restype = c_int
fscanf.argtypes = [POINTER(FILE), STRING]
fseek = _libraries[librslPath].fseek
fseek.restype = c_int
fseek.argtypes = [POINTER(FILE), c_long, c_int]
fsetpos = _libraries[librslPath].fsetpos
fsetpos.restype = c_int
fsetpos.argtypes = [POINTER(FILE), POINTER(fpos_t)]
ftell = _libraries[librslPath].ftell
ftell.restype = c_long
ftell.argtypes = [POINTER(FILE)]
fwrite = _libraries[librslPath].fwrite
fwrite.restype = size_t
fwrite.argtypes = [c_void_p, size_t, size_t, POINTER(FILE)]
getc = _libraries[librslPath].getc
getc.restype = c_int
getc.argtypes = [POINTER(FILE)]
getchar = _libraries[librslPath].getchar
getchar.restype = c_int
getchar.argtypes = []
gets = _libraries[librslPath].gets
gets.restype = STRING
gets.argtypes = [STRING]
perror = _libraries[librslPath].perror
perror.restype = None
perror.argtypes = [STRING]
printf = _libraries[librslPath].printf
printf.restype = c_int
printf.argtypes = [STRING]
putc = _libraries[librslPath].putc
putc.restype = c_int
putc.argtypes = [c_int, POINTER(FILE)]
putchar = _libraries[librslPath].putchar
putchar.restype = c_int
putchar.argtypes = [c_int]
puts = _libraries[librslPath].puts
puts.restype = c_int
puts.argtypes = [STRING]
remove = _libraries[librslPath].remove
remove.restype = c_int
remove.argtypes = [STRING]
rename = _libraries[librslPath].rename
rename.restype = c_int
rename.argtypes = [STRING, STRING]
rewind = _libraries[librslPath].rewind
rewind.restype = None
rewind.argtypes = [POINTER(FILE)]
scanf = _libraries[librslPath].scanf
scanf.restype = c_int
scanf.argtypes = [STRING]
setbuf = _libraries[librslPath].setbuf
setbuf.restype = None
setbuf.argtypes = [POINTER(FILE), STRING]
setvbuf = _libraries[librslPath].setvbuf
setvbuf.restype = c_int
setvbuf.argtypes = [POINTER(FILE), STRING, c_int, size_t]
sprintf = _libraries[librslPath].sprintf
sprintf.restype = c_int
sprintf.argtypes = [STRING, STRING]
sscanf = _libraries[librslPath].sscanf
sscanf.restype = c_int
sscanf.argtypes = [STRING, STRING]
tmpfile = _libraries[librslPath].tmpfile
tmpfile.restype = POINTER(FILE)
tmpfile.argtypes = []
tmpnam = _libraries[librslPath].tmpnam
tmpnam.restype = STRING
tmpnam.argtypes = [STRING]
ungetc = _libraries[librslPath].ungetc
ungetc.restype = c_int
ungetc.argtypes = [c_int, POINTER(FILE)]
vfprintf = _libraries[librslPath].vfprintf
vfprintf.restype = c_int
vfprintf.argtypes = [POINTER(FILE), STRING, STRING]
vprintf = _libraries[librslPath].vprintf
vprintf.restype = c_int
vprintf.argtypes = [STRING, STRING]
vsprintf = _libraries[librslPath].vsprintf
vsprintf.restype = c_int
vsprintf.argtypes = [STRING, STRING, STRING]
asprintf = _libraries[librslPath].asprintf
asprintf.restype = c_int
asprintf.argtypes = [POINTER(STRING), STRING]
vasprintf = _libraries[librslPath].vasprintf
vasprintf.restype = c_int
vasprintf.argtypes = [POINTER(STRING), STRING, STRING]
ctermid = _libraries[librslPath].ctermid
ctermid.restype = STRING
ctermid.argtypes = [STRING]
ctermid_r = _libraries[librslPath].ctermid_r
ctermid_r.restype = STRING
ctermid_r.argtypes = [STRING]
fdopen = _libraries[librslPath].fdopen
fdopen.restype = POINTER(FILE)
fdopen.argtypes = [c_int, STRING]
fileno = _libraries[librslPath].fileno
fileno.restype = c_int
fileno.argtypes = [POINTER(FILE)]
fgetln = _libraries[librslPath].fgetln
fgetln.restype = STRING
fgetln.argtypes = [POINTER(FILE), POINTER(size_t)]
flockfile = _libraries[librslPath].flockfile
flockfile.restype = None
flockfile.argtypes = [POINTER(FILE)]
fmtcheck = _libraries[librslPath].fmtcheck
fmtcheck.restype = STRING
fmtcheck.argtypes = [STRING, STRING]
fpurge = _libraries[librslPath].fpurge
fpurge.restype = c_int
fpurge.argtypes = [POINTER(FILE)]
fseeko = _libraries[librslPath].fseeko
fseeko.restype = c_int
fseeko.argtypes = [POINTER(FILE), fpos_t, c_int]
ftello = _libraries[librslPath].ftello
ftello.restype = fpos_t
ftello.argtypes = [POINTER(FILE)]
ftrylockfile = _libraries[librslPath].ftrylockfile
ftrylockfile.restype = c_int
ftrylockfile.argtypes = [POINTER(FILE)]
funlockfile = _libraries[librslPath].funlockfile
funlockfile.restype = None
funlockfile.argtypes = [POINTER(FILE)]
getc_unlocked = _libraries[librslPath].getc_unlocked
getc_unlocked.restype = c_int
getc_unlocked.argtypes = [POINTER(FILE)]
getchar_unlocked = _libraries[librslPath].getchar_unlocked
getchar_unlocked.restype = c_int
getchar_unlocked.argtypes = []
getw = _libraries[librslPath].getw
getw.restype = c_int
getw.argtypes = [POINTER(FILE)]
popen = _libraries[librslPath].popen
popen.restype = POINTER(FILE)
popen.argtypes = [STRING, STRING]
putc_unlocked = _libraries[librslPath].putc_unlocked
putc_unlocked.restype = c_int
putc_unlocked.argtypes = [c_int, POINTER(FILE)]
putchar_unlocked = _libraries[librslPath].putchar_unlocked
putchar_unlocked.restype = c_int
putchar_unlocked.argtypes = [c_int]
putw = _libraries[librslPath].putw
putw.restype = c_int
putw.argtypes = [c_int, POINTER(FILE)]
setbuffer = _libraries[librslPath].setbuffer
setbuffer.restype = None
setbuffer.argtypes = [POINTER(FILE), STRING, c_int]
setlinebuf = _libraries[librslPath].setlinebuf
setlinebuf.restype = c_int
setlinebuf.argtypes = [POINTER(FILE)]
tempnam = _libraries[librslPath].tempnam
tempnam.restype = STRING
tempnam.argtypes = [STRING, STRING]
snprintf = _libraries[librslPath].snprintf
snprintf.restype = c_int
snprintf.argtypes = [STRING, size_t, STRING]
vfscanf = _libraries[librslPath].vfscanf
vfscanf.restype = c_int
vfscanf.argtypes = [POINTER(FILE), STRING, STRING]
vsnprintf = _libraries[librslPath].vsnprintf
vsnprintf.restype = c_int
vsnprintf.argtypes = [STRING, size_t, STRING, STRING]
vscanf = _libraries[librslPath].vscanf
vscanf.restype = c_int
vscanf.argtypes = [STRING, STRING]
vsscanf = _libraries[librslPath].vsscanf
vsscanf.restype = c_int
vsscanf.argtypes = [STRING, STRING, STRING]
funopen = _libraries[librslPath].funopen
funopen.restype = POINTER(FILE)
funopen.argtypes = [c_void_p, CFUNCTYPE(c_int, c_void_p, STRING, c_int), CFUNCTYPE(c_int, c_void_p, STRING, c_int), CFUNCTYPE(fpos_t, c_void_p, c_longlong, c_int), CFUNCTYPE(c_int, c_void_p)]
__srget = _libraries[librslPath].__srget
__srget.restype = c_int
__srget.argtypes = [POINTER(FILE)]
__svfscanf = _libraries[librslPath].__svfscanf
__svfscanf.restype = c_int
__svfscanf.argtypes = [POINTER(FILE), STRING, STRING]
__swbuf = _libraries[librslPath].__swbuf
__swbuf.restype = c_int
__swbuf.argtypes = [c_int, POINTER(FILE)]
u_char = c_ubyte
u_short = c_ushort
u_int = c_uint
u_long = c_ulong
ushort = c_ushort
uint = c_uint
u_quad_t = u_int64_t
qaddr_t = POINTER(quad_t)
caddr_t = STRING
daddr_t = int32_t
dev_t = int32_t
fixpt_t = u_int32_t
gid_t = u_int32_t
in_addr_t = u_int32_t
in_port_t = u_int16_t
ino_t = u_int32_t
key_t = c_long
mode_t = u_int16_t
nlink_t = u_int16_t
pid_t = int32_t
rlim_t = quad_t
segsz_t = int32_t
swblk_t = int32_t
uid_t = u_int32_t
useconds_t = u_int32_t
clock_t = c_ulong
ssize_t = c_int
time_t = c_long
fd_mask = int32_t
class fd_set(Structure):
    pass
fd_set._fields_ = [
    ('fds_bits', fd_mask * 32),
]
class _pthread_handler_rec(Structure):
    pass
_pthread_handler_rec._fields_ = [
    ('routine', CFUNCTYPE(None, c_void_p)),
    ('arg', c_void_p),
    ('next', POINTER(_pthread_handler_rec)),
]
class _opaque_pthread_t(Structure):
    pass
_opaque_pthread_t._fields_ = [
    ('sig', c_long),
    ('cleanup_stack', POINTER(_pthread_handler_rec)),
    ('opaque', c_char * 596),
]
pthread_t = POINTER(_opaque_pthread_t)
class _opaque_pthread_attr_t(Structure):
    pass
_opaque_pthread_attr_t._fields_ = [
    ('sig', c_long),
    ('opaque', c_char * 36),
]
pthread_attr_t = _opaque_pthread_attr_t
class _opaque_pthread_mutexattr_t(Structure):
    pass
pthread_mutexattr_t = _opaque_pthread_mutexattr_t
_opaque_pthread_mutexattr_t._fields_ = [
    ('sig', c_long),
    ('opaque', c_char * 8),
]
class _opaque_pthread_mutex_t(Structure):
    pass
pthread_mutex_t = _opaque_pthread_mutex_t
_opaque_pthread_mutex_t._fields_ = [
    ('sig', c_long),
    ('opaque', c_char * 40),
]
class _opaque_pthread_condattr_t(Structure):
    pass
pthread_condattr_t = _opaque_pthread_condattr_t
_opaque_pthread_condattr_t._fields_ = [
    ('sig', c_long),
    ('opaque', c_char * 4),
]
class _opaque_pthread_cond_t(Structure):
    pass
_opaque_pthread_cond_t._fields_ = [
    ('sig', c_long),
    ('opaque', c_char * 24),
]
pthread_cond_t = _opaque_pthread_cond_t
class _opaque_pthread_rwlockattr_t(Structure):
    pass
_opaque_pthread_rwlockattr_t._fields_ = [
    ('sig', c_long),
    ('opaque', c_char * 12),
]
pthread_rwlockattr_t = _opaque_pthread_rwlockattr_t
class _opaque_pthread_rwlock_t(Structure):
    pass
_opaque_pthread_rwlock_t._fields_ = [
    ('sig', c_long),
    ('opaque', c_char * 124),
]
pthread_rwlock_t = _opaque_pthread_rwlock_t
class pthread_once_t(Structure):
    pass
pthread_once_t._fields_ = [
    ('sig', c_long),
    ('opaque', c_char * 4),
]
pthread_key_t = c_ulong

# values for enumeration 'Rsl_magic_num'
Rsl_magic_num = c_int # enum

# values for enumeration 'File_type'
File_type = c_int # enum
Range = c_ushort
class Ray_header(Structure):
    pass
Ray_header._fields_ = [
    ('month', c_int),
    ('day', c_int),
    ('year', c_int),
    ('hour', c_int),
    ('minute', c_int),
    ('sec', c_float),
    ('unam_rng', c_float),
    ('azimuth', c_float),
    ('ray_num', c_int),
    ('elev', c_float),
    ('elev_num', c_int),
    ('range_bin1', c_int),
    ('gate_size', c_int),
    ('vel_res', c_float),
    ('sweep_rate', c_float),
    ('prf', c_int),
    ('azim_rate', c_float),
    ('fix_angle', c_float),
    ('pitch', c_float),
    ('roll', c_float),
    ('heading', c_float),
    ('pitch_rate', c_float),
    ('roll_rate', c_float),
    ('heading_rate', c_float),
    ('lat', c_float),
    ('lon', c_float),
    ('alt', c_int),
    ('rvc', c_float),
    ('vel_east', c_float),
    ('vel_north', c_float),
    ('vel_up', c_float),
    ('pulse_count', c_int),
    ('pulse_width', c_float),
    ('beam_width', c_float),
    ('frequency', c_float),
    ('wavelength', c_float),
    ('nyq_vel', c_float),
    ('f', CFUNCTYPE(c_float, c_ushort)),
    ('invf', CFUNCTYPE(Range, c_float)),
    ('nbins', c_int),
]
class Ray(Structure):
    #pass
    def __getattr__(self,attr):
        if attr=='data':
            nbins = self.h.nbins
            converter = self.h.f
            data = N.array([converter(self.range[i]) for i in range(nbins)])
            return data
        elif attr == 'dists':
            #return range gates in km
            dists = self.h.range_bin1 + self.h.gate_size * N.arange(self.h.nbins, dtype=float) 
            #dists = [RSL_get_range_of_range_index(ray, i) for i in range(ray.nbins)] --- in km
            return dists
        else:
            return self.h.__getattribute__(attr)
Ray._fields_ = [
    ('h', Ray_header),
    ('range', POINTER(Range)),
]
class _azimuth_hash(Structure):
    pass
_azimuth_hash._fields_ = [
    ('ray', POINTER(Ray)),
    ('next', POINTER(_azimuth_hash)),
    ('ray_high', POINTER(_azimuth_hash)),
    ('ray_low', POINTER(_azimuth_hash)),
]
Azimuth_hash = _azimuth_hash
class Hash_table(Structure):
    pass
Hash_table._fields_ = [
    ('indexes', POINTER(POINTER(Azimuth_hash))),
    ('nindexes', c_int),
]
class Sweep_header(Structure):
    pass
Sweep_header._fields_ = [
    ('sweep_num', c_int),
    ('elev', c_float),
    ('beam_width', c_float),
    ('vert_half_bw', c_float),
    ('horz_half_bw', c_float),
    ('nrays', c_int),
    ('f', CFUNCTYPE(c_float, c_ushort)),
    ('invf', CFUNCTYPE(Range, c_float)),
]
class Sweep(Structure):
    #pass
    def __getattr__(self,attr):
        if attr=='rays':
            nrays = self.h.nrays
            rays = []
            for i in range(nrays):
                #There are null pointers where rays don't exist, which throws an error on access
                try: rays.append(self.ray[i].contents)
                except: rays.append(None)
            return rays
        else:
            return self.h.__getattribute__(attr)
Sweep._fields_ = [
    ('h', Sweep_header),
    ('ray', POINTER(POINTER(Ray))),
]
class Volume_header(Structure):
    pass
Volume_header._fields_ = [
    ('type_str', STRING),
    ('nsweeps', c_int),
    ('calibr_const', c_float),
    ('f', CFUNCTYPE(c_float, c_ushort)),
    ('invf', CFUNCTYPE(Range, c_float)),
]
class Volume(Structure):
    #pass
    def __getattr__(self,attr):
        if attr=='sweeps':
            nsweeps = self.h.nsweeps
            swps = []
            for i in range(nsweeps):
                #There are null pointers where sweeps don't exist, which throws an error on access
                try: swps.append(self.sweep[i].contents)
                except: swps.append(None)
            return swps
        else:
            return self.h.__getattribute__(attr)
Volume._fields_ = [
    ('h', Volume_header),
    ('sweep', POINTER(POINTER(Sweep))),
]
Carpi_value = Range
Cappi_value = Range
class Carpi(Structure):
    pass
Carpi._fields_ = [
    ('month', c_int),
    ('day', c_int),
    ('year', c_int),
    ('hour', c_int),
    ('minute', c_int),
    ('sec', c_float),
    ('dx', c_float),
    ('dy', c_float),
    ('nx', c_int),
    ('ny', c_int),
    ('radar_x', c_int),
    ('radar_y', c_int),
    ('height', c_float),
    ('lat', c_float),
    ('lon', c_float),
    ('radar_type', c_char * 50),
    ('field_type', c_int),
    ('interp_method', c_int),
    ('f', CFUNCTYPE(c_float, c_ushort)),
    ('invf', CFUNCTYPE(Carpi_value, c_float)),
    ('data', POINTER(POINTER(Carpi_value))),
]
class Er_loc(Structure):
    pass
Er_loc._fields_ = [
    ('elev', c_float),
    ('srange', c_float),
]
class Cappi(Structure):
    pass
Cappi._fields_ = [
    ('month', c_int),
    ('day', c_int),
    ('year', c_int),
    ('hour', c_int),
    ('minute', c_int),
    ('sec', c_float),
    ('height', c_float),
    ('lat', c_float),
    ('lon', c_float),
    ('field_type', c_int),
    ('radar_type', c_char * 50),
    ('interp_method', c_int),
    ('loc', POINTER(Er_loc)),
    ('sweep', POINTER(Sweep)),
]
Cube_value = Range
Slice_value = Range
class Cube(Structure):
    pass
Cube._fields_ = [
    ('lat', c_float),
    ('lon', c_float),
    ('dx', c_float),
    ('dy', c_float),
    ('dz', c_float),
    ('nx', c_int),
    ('ny', c_int),
    ('nz', c_int),
    ('data_type', STRING),
    ('carpi', POINTER(POINTER(Carpi))),
]
class Slice(Structure):
    pass
Slice._fields_ = [
    ('dx', c_float),
    ('dy', c_float),
    ('nx', c_int),
    ('ny', c_int),
    ('data_type', STRING),
    ('f', CFUNCTYPE(c_float, c_ushort)),
    ('invf', CFUNCTYPE(Slice_value, c_float)),
    ('data', POINTER(POINTER(Slice_value))),
]
class Histogram(Structure):
    pass
Histogram._fields_ = [
    ('nbins', c_int),
    ('low', c_int),
    ('hi', c_int),
    ('ucount', c_int),
    ('ccount', c_int),
    ('data', POINTER(c_int)),
]
class Radar_header(Structure):
    pass
Radar_header._fields_ = [
    ('month', c_int),
    ('day', c_int),
    ('year', c_int),
    ('hour', c_int),
    ('minute', c_int),
    ('sec', c_float),
    ('radar_type', c_char * 50),
    ('nvolumes', c_int),
    ('number', c_int),
    ('name', c_char * 8),
    ('radar_name', c_char * 8),
    ('project', c_char * 24),
    ('city', c_char * 15),
    ('state', c_char * 3),
    ('country', c_char * 15),
    ('latd', c_int),
    ('latm', c_int),
    ('lats', c_int),
    ('lond', c_int),
    ('lonm', c_int),
    ('lons', c_int),
    ('height', c_int),
    ('spulse', c_int),
    ('lpulse', c_int),
]
class Radar(Structure):
    #pass
    def __getattr__(self,attr):
        if attr=='volumes':
            nvolumes = self.h.nvolumes
            vols = []
            for i in range(nvolumes):
                #There are null pointers where volumes don't exist, which throws an error on access
                try: vols.append(self.v[i].contents)
                except: vols.append(None)
                #vols = [self.v[i].contents for i in range(nvolumes)]
            return vols
        else:
            return self.h.__getattribute__(attr)
Radar._fields_ = [
    ('h', Radar_header),
    ('v', POINTER(POINTER(Volume))),
]
RSL_africa_to_radar = _libraries[librslPath].RSL_africa_to_radar
RSL_africa_to_radar.restype = POINTER(Radar)
RSL_africa_to_radar.argtypes = [STRING]
RSL_anyformat_to_radar = _libraries[librslPath].RSL_anyformat_to_radar
RSL_anyformat_to_radar.restype = POINTER(Radar)
RSL_anyformat_to_radar.argtypes = [STRING]
RSL_dorade_to_radar = _libraries[librslPath].RSL_dorade_to_radar
RSL_dorade_to_radar.restype = POINTER(Radar)
RSL_dorade_to_radar.argtypes = [STRING]
RSL_EDGE_to_radar = _libraries[librslPath].RSL_EDGE_to_radar
RSL_EDGE_to_radar.restype = POINTER(Radar)
RSL_EDGE_to_radar.argtypes = [STRING]
RSL_fix_radar_header = _libraries[librslPath].RSL_fix_radar_header
RSL_fix_radar_header.restype = POINTER(Radar)
RSL_fix_radar_header.argtypes = [POINTER(Radar)]
RSL_get_window_from_radar = _libraries[librslPath].RSL_get_window_from_radar
RSL_get_window_from_radar.restype = POINTER(Radar)
RSL_get_window_from_radar.argtypes = [POINTER(Radar), c_float, c_float, c_float, c_float]
RSL_hdf_to_radar = _libraries[librslPath].RSL_hdf_to_radar
RSL_hdf_to_radar.restype = POINTER(Radar)
RSL_hdf_to_radar.argtypes = [STRING]
RSL_lassen_to_radar = _libraries[librslPath].RSL_lassen_to_radar
RSL_lassen_to_radar.restype = POINTER(Radar)
RSL_lassen_to_radar.argtypes = [STRING]
RSL_mcgill_to_radar = _libraries[librslPath].RSL_mcgill_to_radar
RSL_mcgill_to_radar.restype = POINTER(Radar)
RSL_mcgill_to_radar.argtypes = [STRING]
RSL_new_radar = _libraries[librslPath].RSL_new_radar
RSL_new_radar.restype = POINTER(Radar)
RSL_new_radar.argtypes = [c_int]
RSL_nsig_to_radar = _libraries[librslPath].RSL_nsig_to_radar
RSL_nsig_to_radar.restype = POINTER(Radar)
RSL_nsig_to_radar.argtypes = [STRING]
RSL_nsig2_to_radar = _libraries[librslPath].RSL_nsig2_to_radar
RSL_nsig2_to_radar.restype = POINTER(Radar)
RSL_nsig2_to_radar.argtypes = [STRING]
RSL_prune_radar = _libraries[librslPath].RSL_prune_radar
RSL_prune_radar.restype = POINTER(Radar)
RSL_prune_radar.argtypes = [POINTER(Radar)]
RSL_radtec_to_radar = _libraries[librslPath].RSL_radtec_to_radar
RSL_radtec_to_radar.restype = POINTER(Radar)
RSL_radtec_to_radar.argtypes = [STRING]
RSL_rainbow_to_radar = _libraries[librslPath].RSL_rainbow_to_radar
RSL_rainbow_to_radar.restype = POINTER(Radar)
RSL_rainbow_to_radar.argtypes = [STRING]
RSL_rapic_to_radar = _libraries[librslPath].RSL_rapic_to_radar
RSL_rapic_to_radar.restype = POINTER(Radar)
RSL_rapic_to_radar.argtypes = [STRING]
RSL_read_radar = _libraries[librslPath].RSL_read_radar
RSL_read_radar.restype = POINTER(Radar)
RSL_read_radar.argtypes = [STRING]
RSL_sort_radar = _libraries[librslPath].RSL_sort_radar
RSL_sort_radar.restype = POINTER(Radar)
RSL_sort_radar.argtypes = [POINTER(Radar)]
RSL_toga_to_radar = _libraries[librslPath].RSL_toga_to_radar
RSL_toga_to_radar.restype = POINTER(Radar)
RSL_toga_to_radar.argtypes = [STRING]
RSL_uf_to_radar = _libraries[librslPath].RSL_uf_to_radar
RSL_uf_to_radar.restype = POINTER(Radar)
RSL_uf_to_radar.argtypes = [STRING]
RSL_uf_to_radar_fp = _libraries[librslPath].RSL_uf_to_radar_fp
RSL_uf_to_radar_fp.restype = POINTER(Radar)
RSL_uf_to_radar_fp.argtypes = [POINTER(FILE)]
RSL_wsr88d_to_radar = _libraries[librslPath].RSL_wsr88d_to_radar
RSL_wsr88d_to_radar.restype = POINTER(Radar)
RSL_wsr88d_to_radar.argtypes = [STRING, STRING]
RSL_clear_volume = _libraries[librslPath].RSL_clear_volume
RSL_clear_volume.restype = POINTER(Volume)
RSL_clear_volume.argtypes = [POINTER(Volume)]
RSL_copy_volume = _libraries[librslPath].RSL_copy_volume
RSL_copy_volume.restype = POINTER(Volume)
RSL_copy_volume.argtypes = [POINTER(Volume)]
RSL_fix_volume_header = _libraries[librslPath].RSL_fix_volume_header
RSL_fix_volume_header.restype = POINTER(Volume)
RSL_fix_volume_header.argtypes = [POINTER(Volume)]
RSL_get_volume = _libraries[librslPath].RSL_get_volume
RSL_get_volume.restype = POINTER(Volume)
RSL_get_volume.argtypes = [POINTER(Radar), c_int]
RSL_get_window_from_volume = _libraries[librslPath].RSL_get_window_from_volume
RSL_get_window_from_volume.restype = POINTER(Volume)
RSL_get_window_from_volume.argtypes = [POINTER(Volume), c_float, c_float, c_float, c_float]
RSL_new_volume = _libraries[librslPath].RSL_new_volume
RSL_new_volume.restype = POINTER(Volume)
RSL_new_volume.argtypes = [c_int]
RSL_prune_volume = _libraries[librslPath].RSL_prune_volume
RSL_prune_volume.restype = POINTER(Volume)
RSL_prune_volume.argtypes = [POINTER(Volume)]
RSL_read_volume = _libraries[librslPath].RSL_read_volume
RSL_read_volume.restype = POINTER(Volume)
RSL_read_volume.argtypes = [POINTER(FILE)]
RSL_reverse_sweep_order = _libraries[librslPath].RSL_reverse_sweep_order
RSL_reverse_sweep_order.restype = POINTER(Volume)
RSL_reverse_sweep_order.argtypes = [POINTER(Volume)]
RSL_sort_rays_in_volume = _libraries[librslPath].RSL_sort_rays_in_volume
RSL_sort_rays_in_volume.restype = POINTER(Volume)
RSL_sort_rays_in_volume.argtypes = [POINTER(Volume)]
RSL_sort_sweeps_in_volume = _libraries[librslPath].RSL_sort_sweeps_in_volume
RSL_sort_sweeps_in_volume.restype = POINTER(Volume)
RSL_sort_sweeps_in_volume.argtypes = [POINTER(Volume)]
RSL_sort_volume = _libraries[librslPath].RSL_sort_volume
RSL_sort_volume.restype = POINTER(Volume)
RSL_sort_volume.argtypes = [POINTER(Volume)]
RSL_volume_z_to_r = _libraries[librslPath].RSL_volume_z_to_r
RSL_volume_z_to_r.restype = POINTER(Volume)
RSL_volume_z_to_r.argtypes = [POINTER(Volume), c_float, c_float]
RSL_clear_sweep = _libraries[librslPath].RSL_clear_sweep
RSL_clear_sweep.restype = POINTER(Sweep)
RSL_clear_sweep.argtypes = [POINTER(Sweep)]
RSL_copy_sweep = _libraries[librslPath].RSL_copy_sweep
RSL_copy_sweep.restype = POINTER(Sweep)
RSL_copy_sweep.argtypes = [POINTER(Sweep)]
RSL_fix_sweep_header = _libraries[librslPath].RSL_fix_sweep_header
RSL_fix_sweep_header.restype = POINTER(Sweep)
RSL_fix_sweep_header.argtypes = [POINTER(Sweep)]
RSL_get_closest_sweep = _libraries[librslPath].RSL_get_closest_sweep
RSL_get_closest_sweep.restype = POINTER(Sweep)
RSL_get_closest_sweep.argtypes = [POINTER(Volume), c_float, c_float]
RSL_get_first_sweep_of_volume = _libraries[librslPath].RSL_get_first_sweep_of_volume
RSL_get_first_sweep_of_volume.restype = POINTER(Sweep)
RSL_get_first_sweep_of_volume.argtypes = [POINTER(Volume)]
RSL_get_sweep = _libraries[librslPath].RSL_get_sweep
RSL_get_sweep.restype = POINTER(Sweep)
RSL_get_sweep.argtypes = [POINTER(Volume), c_float]
RSL_get_window_from_sweep = _libraries[librslPath].RSL_get_window_from_sweep
RSL_get_window_from_sweep.restype = POINTER(Sweep)
RSL_get_window_from_sweep.argtypes = [POINTER(Sweep), c_float, c_float, c_float, c_float]
RSL_new_sweep = _libraries[librslPath].RSL_new_sweep
RSL_new_sweep.restype = POINTER(Sweep)
RSL_new_sweep.argtypes = [c_int]
RSL_prune_sweep = _libraries[librslPath].RSL_prune_sweep
RSL_prune_sweep.restype = POINTER(Sweep)
RSL_prune_sweep.argtypes = [POINTER(Sweep)]
RSL_read_sweep = _libraries[librslPath].RSL_read_sweep
RSL_read_sweep.restype = POINTER(Sweep)
RSL_read_sweep.argtypes = [POINTER(FILE)]
RSL_sort_rays_in_sweep = _libraries[librslPath].RSL_sort_rays_in_sweep
RSL_sort_rays_in_sweep.restype = POINTER(Sweep)
RSL_sort_rays_in_sweep.argtypes = [POINTER(Sweep)]
RSL_sort_rays_by_time = _libraries[librslPath].RSL_sort_rays_by_time
RSL_sort_rays_by_time.restype = POINTER(Sweep)
RSL_sort_rays_by_time.argtypes = [POINTER(Sweep)]
RSL_sweep_z_to_r = _libraries[librslPath].RSL_sweep_z_to_r
RSL_sweep_z_to_r.restype = POINTER(Sweep)
RSL_sweep_z_to_r.argtypes = [POINTER(Sweep), c_float, c_float]
RSL_clear_ray = _libraries[librslPath].RSL_clear_ray
RSL_clear_ray.restype = POINTER(Ray)
RSL_clear_ray.argtypes = [POINTER(Ray)]
RSL_copy_ray = _libraries[librslPath].RSL_copy_ray
RSL_copy_ray.restype = POINTER(Ray)
RSL_copy_ray.argtypes = [POINTER(Ray)]
RSL_get_first_ray_of_sweep = _libraries[librslPath].RSL_get_first_ray_of_sweep
RSL_get_first_ray_of_sweep.restype = POINTER(Ray)
RSL_get_first_ray_of_sweep.argtypes = [POINTER(Sweep)]
RSL_get_first_ray_of_volume = _libraries[librslPath].RSL_get_first_ray_of_volume
RSL_get_first_ray_of_volume.restype = POINTER(Ray)
RSL_get_first_ray_of_volume.argtypes = [POINTER(Volume)]
RSL_get_closest_ray_from_sweep = _libraries[librslPath].RSL_get_closest_ray_from_sweep
RSL_get_closest_ray_from_sweep.restype = POINTER(Ray)
RSL_get_closest_ray_from_sweep.argtypes = [POINTER(Sweep), c_float, c_float]
RSL_get_next_ccwise_ray = _libraries[librslPath].RSL_get_next_ccwise_ray
RSL_get_next_ccwise_ray.restype = POINTER(Ray)
RSL_get_next_ccwise_ray.argtypes = [POINTER(Sweep), POINTER(Ray)]
RSL_get_next_cwise_ray = _libraries[librslPath].RSL_get_next_cwise_ray
RSL_get_next_cwise_ray.restype = POINTER(Ray)
RSL_get_next_cwise_ray.argtypes = [POINTER(Sweep), POINTER(Ray)]
RSL_get_ray = _libraries[librslPath].RSL_get_ray
RSL_get_ray.restype = POINTER(Ray)
RSL_get_ray.argtypes = [POINTER(Volume), c_float, c_float]
RSL_get_ray_above = _libraries[librslPath].RSL_get_ray_above
RSL_get_ray_above.restype = POINTER(Ray)
RSL_get_ray_above.argtypes = [POINTER(Volume), POINTER(Ray)]
RSL_get_ray_below = _libraries[librslPath].RSL_get_ray_below
RSL_get_ray_below.restype = POINTER(Ray)
RSL_get_ray_below.argtypes = [POINTER(Volume), POINTER(Ray)]
RSL_get_ray_from_sweep = _libraries[librslPath].RSL_get_ray_from_sweep
RSL_get_ray_from_sweep.restype = POINTER(Ray)
RSL_get_ray_from_sweep.argtypes = [POINTER(Sweep), c_float]
RSL_get_window_from_ray = _libraries[librslPath].RSL_get_window_from_ray
RSL_get_window_from_ray.restype = POINTER(Ray)
RSL_get_window_from_ray.argtypes = [POINTER(Ray), c_float, c_float, c_float, c_float]
RSL_new_ray = _libraries[librslPath].RSL_new_ray
RSL_new_ray.restype = POINTER(Ray)
RSL_new_ray.argtypes = [c_int]
RSL_prune_ray = _libraries[librslPath].RSL_prune_ray
RSL_prune_ray.restype = POINTER(Ray)
RSL_prune_ray.argtypes = [POINTER(Ray)]
RSL_ray_z_to_r = _libraries[librslPath].RSL_ray_z_to_r
RSL_ray_z_to_r.restype = POINTER(Ray)
RSL_ray_z_to_r.argtypes = [POINTER(Ray), c_float, c_float]
RSL_read_ray = _libraries[librslPath].RSL_read_ray
RSL_read_ray.restype = POINTER(Ray)
RSL_read_ray.argtypes = [POINTER(FILE)]
RSL_area_of_ray = _libraries[librslPath].RSL_area_of_ray
RSL_area_of_ray.restype = c_float
RSL_area_of_ray.argtypes = [POINTER(Ray), c_float, c_float, c_float, c_float]
RSL_fraction_of_ray = _libraries[librslPath].RSL_fraction_of_ray
RSL_fraction_of_ray.restype = c_float
RSL_fraction_of_ray.argtypes = [POINTER(Ray), c_float, c_float, c_float]
RSL_fraction_of_sweep = _libraries[librslPath].RSL_fraction_of_sweep
RSL_fraction_of_sweep.restype = c_float
RSL_fraction_of_sweep.argtypes = [POINTER(Sweep), c_float, c_float, c_float]
RSL_fraction_of_volume = _libraries[librslPath].RSL_fraction_of_volume
RSL_fraction_of_volume.restype = c_float
RSL_fraction_of_volume.argtypes = [POINTER(Volume), c_float, c_float, c_float]
RSL_fractional_area_of_sweep = _libraries[librslPath].RSL_fractional_area_of_sweep
RSL_fractional_area_of_sweep.restype = c_float
RSL_fractional_area_of_sweep.argtypes = [POINTER(Sweep), c_float, c_float, c_float, c_float]
RSL_get_linear_value = _libraries[librslPath].RSL_get_linear_value
RSL_get_linear_value.restype = c_float
RSL_get_linear_value.argtypes = [POINTER(Volume), c_float, c_float, c_float, c_float]
RSL_get_nyquist_from_radar = _libraries[librslPath].RSL_get_nyquist_from_radar
RSL_get_nyquist_from_radar.restype = c_float
RSL_get_nyquist_from_radar.argtypes = [POINTER(Radar)]
RSL_get_range_of_range_index = _libraries[librslPath].RSL_get_range_of_range_index
RSL_get_range_of_range_index.restype = c_float
RSL_get_range_of_range_index.argtypes = [POINTER(Ray), c_int]
RSL_get_value = _libraries[librslPath].RSL_get_value
RSL_get_value.restype = c_float
RSL_get_value.argtypes = [POINTER(Volume), c_float, c_float, c_float]
RSL_get_value_at_h = _libraries[librslPath].RSL_get_value_at_h
RSL_get_value_at_h.restype = c_float
RSL_get_value_at_h.argtypes = [POINTER(Volume), c_float, c_float, c_float]
RSL_get_value_from_cappi = _libraries[librslPath].RSL_get_value_from_cappi
RSL_get_value_from_cappi.restype = c_float
RSL_get_value_from_cappi.argtypes = [POINTER(Cappi), c_float, c_float]
RSL_get_value_from_ray = _libraries[librslPath].RSL_get_value_from_ray
RSL_get_value_from_ray.restype = c_float
RSL_get_value_from_ray.argtypes = [POINTER(Ray), c_float]
RSL_get_value_from_sweep = _libraries[librslPath].RSL_get_value_from_sweep
RSL_get_value_from_sweep.restype = c_float
RSL_get_value_from_sweep.argtypes = [POINTER(Sweep), c_float, c_float]
RSL_z_to_r = _libraries[librslPath].RSL_z_to_r
RSL_z_to_r.restype = c_float
RSL_z_to_r.argtypes = [c_float, c_float, c_float]
RSL_fill_cappi = _libraries[librslPath].RSL_fill_cappi
RSL_fill_cappi.restype = c_int
RSL_fill_cappi.argtypes = [POINTER(Volume), POINTER(Cappi), c_int]
RSL_write_histogram = _libraries[librslPath].RSL_write_histogram
RSL_write_histogram.restype = c_int
RSL_write_histogram.argtypes = [POINTER(Histogram), STRING]
RSL_write_ray = _libraries[librslPath].RSL_write_ray
RSL_write_ray.restype = c_int
RSL_write_ray.argtypes = [POINTER(Ray), POINTER(FILE)]
RSL_write_sweep = _libraries[librslPath].RSL_write_sweep
RSL_write_sweep.restype = c_int
RSL_write_sweep.argtypes = [POINTER(Sweep), POINTER(FILE)]
RSL_write_radar = _libraries[librslPath].RSL_write_radar
RSL_write_radar.restype = c_int
RSL_write_radar.argtypes = [POINTER(Radar), STRING]
RSL_write_radar_gzip = _libraries[librslPath].RSL_write_radar_gzip
RSL_write_radar_gzip.restype = c_int
RSL_write_radar_gzip.argtypes = [POINTER(Radar), STRING]
RSL_write_volume = _libraries[librslPath].RSL_write_volume
RSL_write_volume.restype = c_int
RSL_write_volume.argtypes = [POINTER(Volume), POINTER(FILE)]
RSL_rhi_sweep_to_cart = _libraries[librslPath].RSL_rhi_sweep_to_cart
RSL_rhi_sweep_to_cart.restype = POINTER(c_ubyte)
RSL_rhi_sweep_to_cart.argtypes = [POINTER(Sweep), c_int, c_int, c_float, c_int]
RSL_sweep_to_cart = _libraries[librslPath].RSL_sweep_to_cart
RSL_sweep_to_cart.restype = POINTER(c_ubyte)
RSL_sweep_to_cart.argtypes = [POINTER(Sweep), c_int, c_int, c_float]
RSL_add_dbz_offset_to_ray = _libraries[librslPath].RSL_add_dbz_offset_to_ray
RSL_add_dbz_offset_to_ray.restype = None
RSL_add_dbz_offset_to_ray.argtypes = [POINTER(Ray), c_float]
RSL_add_dbz_offset_to_sweep = _libraries[librslPath].RSL_add_dbz_offset_to_sweep
RSL_add_dbz_offset_to_sweep.restype = None
RSL_add_dbz_offset_to_sweep.argtypes = [POINTER(Sweep), c_float]
RSL_add_dbz_offset_to_volume = _libraries[librslPath].RSL_add_dbz_offset_to_volume
RSL_add_dbz_offset_to_volume.restype = None
RSL_add_dbz_offset_to_volume.argtypes = [POINTER(Volume), c_float]
RSL_bscan_ray = _libraries[librslPath].RSL_bscan_ray
RSL_bscan_ray.restype = None
RSL_bscan_ray.argtypes = [POINTER(Ray), POINTER(FILE)]
RSL_bscan_sweep = _libraries[librslPath].RSL_bscan_sweep
RSL_bscan_sweep.restype = None
RSL_bscan_sweep.argtypes = [POINTER(Sweep), STRING]
RSL_bscan_volume = _libraries[librslPath].RSL_bscan_volume
RSL_bscan_volume.restype = None
RSL_bscan_volume.argtypes = [POINTER(Volume), STRING]
RSL_find_rng_azm = _libraries[librslPath].RSL_find_rng_azm
RSL_find_rng_azm.restype = None
RSL_find_rng_azm.argtypes = [POINTER(c_float), POINTER(c_float), c_float, c_float]
RSL_fix_time = _libraries[librslPath].RSL_fix_time
RSL_fix_time.restype = None
RSL_fix_time.argtypes = [POINTER(Ray)]
RSL_free_cappi = _libraries[librslPath].RSL_free_cappi
RSL_free_cappi.restype = None
RSL_free_cappi.argtypes = [POINTER(Cappi)]
RSL_free_carpi = _libraries[librslPath].RSL_free_carpi
RSL_free_carpi.restype = None
RSL_free_carpi.argtypes = [POINTER(Carpi)]
RSL_free_cube = _libraries[librslPath].RSL_free_cube
RSL_free_cube.restype = None
RSL_free_cube.argtypes = [POINTER(Cube)]
RSL_free_histogram = _libraries[librslPath].RSL_free_histogram
RSL_free_histogram.restype = None
RSL_free_histogram.argtypes = [POINTER(Histogram)]
RSL_free_ray = _libraries[librslPath].RSL_free_ray
RSL_free_ray.restype = None
RSL_free_ray.argtypes = [POINTER(Ray)]
RSL_free_slice = _libraries[librslPath].RSL_free_slice
RSL_free_slice.restype = None
RSL_free_slice.argtypes = [POINTER(Slice)]
RSL_free_sweep = _libraries[librslPath].RSL_free_sweep
RSL_free_sweep.restype = None
RSL_free_sweep.argtypes = [POINTER(Sweep)]
RSL_free_radar = _libraries[librslPath].RSL_free_radar
RSL_free_radar.restype = None
RSL_free_radar.argtypes = [POINTER(Radar)]
RSL_free_volume = _libraries[librslPath].RSL_free_volume
RSL_free_volume.restype = None
RSL_free_volume.argtypes = [POINTER(Volume)]
RSL_get_color_table = _libraries[librslPath].RSL_get_color_table
RSL_get_color_table.restype = None
RSL_get_color_table.argtypes = [c_int, STRING, POINTER(c_int)]
RSL_get_groundr_and_h = _libraries[librslPath].RSL_get_groundr_and_h
RSL_get_groundr_and_h.restype = None
RSL_get_groundr_and_h.argtypes = [c_float, c_float, POINTER(c_float), POINTER(c_float)]
RSL_get_slantr_and_elev = _libraries[librslPath].RSL_get_slantr_and_elev
RSL_get_slantr_and_elev.restype = None
RSL_get_slantr_and_elev.argtypes = [c_float, c_float, POINTER(c_float), POINTER(c_float)]
RSL_get_slantr_and_h = _libraries[librslPath].RSL_get_slantr_and_h
RSL_get_slantr_and_h.restype = None
RSL_get_slantr_and_h.argtypes = [c_float, c_float, POINTER(c_float), POINTER(c_float)]
RSL_load_color_table = _libraries[librslPath].RSL_load_color_table
RSL_load_color_table.restype = None
RSL_load_color_table.argtypes = [STRING, STRING, POINTER(c_int)]
RSL_load_height_color_table = _libraries[librslPath].RSL_load_height_color_table
RSL_load_height_color_table.restype = None
RSL_load_height_color_table.argtypes = []
RSL_load_rainfall_color_table = _libraries[librslPath].RSL_load_rainfall_color_table
RSL_load_rainfall_color_table.restype = None
RSL_load_rainfall_color_table.argtypes = []
RSL_load_refl_color_table = _libraries[librslPath].RSL_load_refl_color_table
RSL_load_refl_color_table.restype = None
RSL_load_refl_color_table.argtypes = []
RSL_load_vel_color_table = _libraries[librslPath].RSL_load_vel_color_table
RSL_load_vel_color_table.restype = None
RSL_load_vel_color_table.argtypes = []
RSL_load_sw_color_table = _libraries[librslPath].RSL_load_sw_color_table
RSL_load_sw_color_table.restype = None
RSL_load_sw_color_table.argtypes = []
RSL_load_zdr_color_table = _libraries[librslPath].RSL_load_zdr_color_table
RSL_load_zdr_color_table.restype = None
RSL_load_zdr_color_table.argtypes = []
RSL_load_red_table = _libraries[librslPath].RSL_load_red_table
RSL_load_red_table.restype = None
RSL_load_red_table.argtypes = [STRING]
RSL_load_green_table = _libraries[librslPath].RSL_load_green_table
RSL_load_green_table.restype = None
RSL_load_green_table.argtypes = [STRING]
RSL_load_blue_table = _libraries[librslPath].RSL_load_blue_table
RSL_load_blue_table.restype = None
RSL_load_blue_table.argtypes = [STRING]
RSL_print_histogram = _libraries[librslPath].RSL_print_histogram
RSL_print_histogram.restype = None
RSL_print_histogram.argtypes = [POINTER(Histogram), c_int, c_int, STRING]
RSL_print_version = _libraries[librslPath].RSL_print_version
RSL_print_version.restype = None
RSL_print_version.argtypes = []
RSL_radar_to_uf = _libraries[librslPath].RSL_radar_to_uf
RSL_radar_to_uf.restype = None
RSL_radar_to_uf.argtypes = [POINTER(Radar), STRING]
RSL_radar_to_uf_gzip = _libraries[librslPath].RSL_radar_to_uf_gzip
RSL_radar_to_uf_gzip.restype = None
RSL_radar_to_uf_gzip.argtypes = [POINTER(Radar), STRING]
RSL_radar_verbose_off = _libraries[librslPath].RSL_radar_verbose_off
RSL_radar_verbose_off.restype = None
RSL_radar_verbose_off.argtypes = []
RSL_radar_verbose_on = _libraries[librslPath].RSL_radar_verbose_on
RSL_radar_verbose_on.restype = None
RSL_radar_verbose_on.argtypes = []
RSL_read_these_sweeps = _libraries[librslPath].RSL_read_these_sweeps
RSL_read_these_sweeps.restype = None
RSL_read_these_sweeps.argtypes = [STRING]
RSL_rebin_velocity_ray = _libraries[librslPath].RSL_rebin_velocity_ray
RSL_rebin_velocity_ray.restype = None
RSL_rebin_velocity_ray.argtypes = [POINTER(Ray)]
RSL_rebin_velocity_sweep = _libraries[librslPath].RSL_rebin_velocity_sweep
RSL_rebin_velocity_sweep.restype = None
RSL_rebin_velocity_sweep.argtypes = [POINTER(Sweep)]
RSL_rebin_velocity_volume = _libraries[librslPath].RSL_rebin_velocity_volume
RSL_rebin_velocity_volume.restype = None
RSL_rebin_velocity_volume.argtypes = [POINTER(Volume)]
RSL_rebin_zdr_ray = _libraries[librslPath].RSL_rebin_zdr_ray
RSL_rebin_zdr_ray.restype = None
RSL_rebin_zdr_ray.argtypes = [POINTER(Ray)]
RSL_rebin_zdr_sweep = _libraries[librslPath].RSL_rebin_zdr_sweep
RSL_rebin_zdr_sweep.restype = None
RSL_rebin_zdr_sweep.argtypes = [POINTER(Sweep)]
RSL_rebin_zdr_volume = _libraries[librslPath].RSL_rebin_zdr_volume
RSL_rebin_zdr_volume.restype = None
RSL_rebin_zdr_volume.argtypes = [POINTER(Volume)]
RSL_rhi_sweep_to_gif = _libraries[librslPath].RSL_rhi_sweep_to_gif
RSL_rhi_sweep_to_gif.restype = None
RSL_rhi_sweep_to_gif.argtypes = [POINTER(Sweep), STRING, c_int, c_int, c_float, c_int]
RSL_select_fields = _libraries[librslPath].RSL_select_fields
RSL_select_fields.restype = None
RSL_select_fields.argtypes = [STRING]
RSL_set_color_table = _libraries[librslPath].RSL_set_color_table
RSL_set_color_table.restype = None
RSL_set_color_table.argtypes = [c_int, STRING, c_int]
RSL_sweep_to_gif = _libraries[librslPath].RSL_sweep_to_gif
RSL_sweep_to_gif.restype = None
RSL_sweep_to_gif.argtypes = [POINTER(Sweep), STRING, c_int, c_int, c_float]
RSL_sweep_to_pgm = _libraries[librslPath].RSL_sweep_to_pgm
RSL_sweep_to_pgm.restype = None
RSL_sweep_to_pgm.argtypes = [POINTER(Sweep), STRING, c_int, c_int, c_float]
RSL_sweep_to_pict = _libraries[librslPath].RSL_sweep_to_pict
RSL_sweep_to_pict.restype = None
RSL_sweep_to_pict.argtypes = [POINTER(Sweep), STRING, c_int, c_int, c_float]
RSL_sweep_to_ppm = _libraries[librslPath].RSL_sweep_to_ppm
RSL_sweep_to_ppm.restype = None
RSL_sweep_to_ppm.argtypes = [POINTER(Sweep), STRING, c_int, c_int, c_float]
RSL_volume_to_gif = _libraries[librslPath].RSL_volume_to_gif
RSL_volume_to_gif.restype = None
RSL_volume_to_gif.argtypes = [POINTER(Volume), STRING, c_int, c_int, c_float]
RSL_volume_to_pgm = _libraries[librslPath].RSL_volume_to_pgm
RSL_volume_to_pgm.restype = None
RSL_volume_to_pgm.argtypes = [POINTER(Volume), STRING, c_int, c_int, c_float]
RSL_volume_to_pict = _libraries[librslPath].RSL_volume_to_pict
RSL_volume_to_pict.restype = None
RSL_volume_to_pict.argtypes = [POINTER(Volume), STRING, c_int, c_int, c_float]
RSL_volume_to_ppm = _libraries[librslPath].RSL_volume_to_ppm
RSL_volume_to_ppm.restype = None
RSL_volume_to_ppm.argtypes = [POINTER(Volume), STRING, c_int, c_int, c_float]
RSL_write_gif = _libraries[librslPath].RSL_write_gif
RSL_write_gif.restype = None
RSL_write_gif.argtypes = [STRING, POINTER(c_ubyte), c_int, c_int, POINTER(c_char * 3)]
RSL_write_pgm = _libraries[librslPath].RSL_write_pgm
RSL_write_pgm.restype = None
RSL_write_pgm.argtypes = [STRING, POINTER(c_ubyte), c_int, c_int]
RSL_write_pict = _libraries[librslPath].RSL_write_pict
RSL_write_pict.restype = None
RSL_write_pict.argtypes = [STRING, POINTER(c_ubyte), c_int, c_int, POINTER(c_char * 3)]
RSL_write_ppm = _libraries[librslPath].RSL_write_ppm
RSL_write_ppm.restype = None
RSL_write_ppm.argtypes = [STRING, POINTER(c_ubyte), c_int, c_int, POINTER(c_char * 3)]
RSL_new_cappi = _libraries[librslPath].RSL_new_cappi
RSL_new_cappi.restype = POINTER(Cappi)
RSL_new_cappi.argtypes = [POINTER(Sweep), c_float]
RSL_cappi_at_h = _libraries[librslPath].RSL_cappi_at_h
RSL_cappi_at_h.restype = POINTER(Cappi)
RSL_cappi_at_h.argtypes = [POINTER(Volume), c_float, c_float]
RSL_cappi_to_carpi = _libraries[librslPath].RSL_cappi_to_carpi
RSL_cappi_to_carpi.restype = POINTER(Carpi)
RSL_cappi_to_carpi.argtypes = [POINTER(Cappi), c_float, c_float, c_float, c_float, c_int, c_int, c_int, c_int]
RSL_new_carpi = _libraries[librslPath].RSL_new_carpi
RSL_new_carpi.restype = POINTER(Carpi)
RSL_new_carpi.argtypes = [c_int, c_int]
RSL_volume_to_carpi = _libraries[librslPath].RSL_volume_to_carpi
RSL_volume_to_carpi.restype = POINTER(Carpi)
RSL_volume_to_carpi.argtypes = [POINTER(Volume), c_float, c_float, c_float, c_float, c_int, c_int, c_int, c_int, c_float, c_float]
RSL_new_cube = _libraries[librslPath].RSL_new_cube
RSL_new_cube.restype = POINTER(Cube)
RSL_new_cube.argtypes = [c_int]
RSL_volume_to_cube = _libraries[librslPath].RSL_volume_to_cube
RSL_volume_to_cube.restype = POINTER(Cube)
RSL_volume_to_cube.argtypes = [POINTER(Volume), c_float, c_float, c_float, c_int, c_int, c_int, c_float, c_int, c_int, c_int]
RSL_new_slice = _libraries[librslPath].RSL_new_slice
RSL_new_slice.restype = POINTER(Slice)
RSL_new_slice.argtypes = [c_int, c_int]
RSL_get_slice_from_cube = _libraries[librslPath].RSL_get_slice_from_cube
RSL_get_slice_from_cube.restype = POINTER(Slice)
RSL_get_slice_from_cube.argtypes = [POINTER(Cube), c_int, c_int, c_int]
RSL_allocate_histogram = _libraries[librslPath].RSL_allocate_histogram
RSL_allocate_histogram.restype = POINTER(Histogram)
RSL_allocate_histogram.argtypes = [c_int, c_int]
RSL_get_histogram_from_ray = _libraries[librslPath].RSL_get_histogram_from_ray
RSL_get_histogram_from_ray.restype = POINTER(Histogram)
RSL_get_histogram_from_ray.argtypes = [POINTER(Ray), POINTER(Histogram), c_int, c_int, c_int, c_int]
RSL_get_histogram_from_sweep = _libraries[librslPath].RSL_get_histogram_from_sweep
RSL_get_histogram_from_sweep.restype = POINTER(Histogram)
RSL_get_histogram_from_sweep.argtypes = [POINTER(Sweep), POINTER(Histogram), c_int, c_int, c_int, c_int]
RSL_get_histogram_from_volume = _libraries[librslPath].RSL_get_histogram_from_volume
RSL_get_histogram_from_volume.restype = POINTER(Histogram)
RSL_get_histogram_from_volume.argtypes = [POINTER(Volume), POINTER(Histogram), c_int, c_int, c_int, c_int]
RSL_read_histogram = _libraries[librslPath].RSL_read_histogram
RSL_read_histogram.restype = POINTER(Histogram)
RSL_read_histogram.argtypes = [STRING]
no_command = _libraries[librslPath].no_command
no_command.restype = c_int
no_command.argtypes = [STRING]
uncompress_pipe = _libraries[librslPath].uncompress_pipe
uncompress_pipe.restype = POINTER(FILE)
uncompress_pipe.argtypes = [POINTER(FILE)]
compress_pipe = _libraries[librslPath].compress_pipe
compress_pipe.restype = POINTER(FILE)
compress_pipe.argtypes = [POINTER(FILE)]
rsl_pclose = _libraries[librslPath].rsl_pclose
rsl_pclose.restype = c_int
rsl_pclose.argtypes = [POINTER(FILE)]
RSL_carpi_to_cart = _libraries[librslPath].RSL_carpi_to_cart
RSL_carpi_to_cart.restype = POINTER(c_ubyte)
RSL_carpi_to_cart.argtypes = [POINTER(Carpi), c_int, c_int, c_float]
RSL_carpi_to_gif = _libraries[librslPath].RSL_carpi_to_gif
RSL_carpi_to_gif.restype = None
RSL_carpi_to_gif.argtypes = [POINTER(Carpi), STRING, c_int, c_int, c_float]
RSL_carpi_to_pict = _libraries[librslPath].RSL_carpi_to_pict
RSL_carpi_to_pict.restype = None
RSL_carpi_to_pict.argtypes = [POINTER(Carpi), STRING, c_int, c_int, c_float]
RSL_carpi_to_ppm = _libraries[librslPath].RSL_carpi_to_ppm
RSL_carpi_to_ppm.restype = None
RSL_carpi_to_ppm.argtypes = [POINTER(Carpi), STRING, c_int, c_int, c_float]
RSL_carpi_to_pgm = _libraries[librslPath].RSL_carpi_to_pgm
RSL_carpi_to_pgm.restype = None
RSL_carpi_to_pgm.argtypes = [POINTER(Carpi), STRING, c_int, c_int, c_float]
DZ_F = _libraries[librslPath].DZ_F
DZ_F.restype = c_float
DZ_F.argtypes = [Range]
VR_F = _libraries[librslPath].VR_F
VR_F.restype = c_float
VR_F.argtypes = [Range]
SW_F = _libraries[librslPath].SW_F
SW_F.restype = c_float
SW_F.argtypes = [Range]
CZ_F = _libraries[librslPath].CZ_F
CZ_F.restype = c_float
CZ_F.argtypes = [Range]
ZT_F = _libraries[librslPath].ZT_F
ZT_F.restype = c_float
ZT_F.argtypes = [Range]
DR_F = _libraries[librslPath].DR_F
DR_F.restype = c_float
DR_F.argtypes = [Range]
LR_F = _libraries[librslPath].LR_F
LR_F.restype = c_float
LR_F.argtypes = [Range]
ZD_F = _libraries[librslPath].ZD_F
ZD_F.restype = c_float
ZD_F.argtypes = [Range]
DM_F = _libraries[librslPath].DM_F
DM_F.restype = c_float
DM_F.argtypes = [Range]
RH_F = _libraries[librslPath].RH_F
RH_F.restype = c_float
RH_F.argtypes = [Range]
PH_F = _libraries[librslPath].PH_F
PH_F.restype = c_float
PH_F.argtypes = [Range]
XZ_F = _libraries[librslPath].XZ_F
XZ_F.restype = c_float
XZ_F.argtypes = [Range]
CD_F = _libraries[librslPath].CD_F
CD_F.restype = c_float
CD_F.argtypes = [Range]
MZ_F = _libraries[librslPath].MZ_F
MZ_F.restype = c_float
MZ_F.argtypes = [Range]
MD_F = _libraries[librslPath].MD_F
MD_F.restype = c_float
MD_F.argtypes = [Range]
ZE_F = _libraries[librslPath].ZE_F
ZE_F.restype = c_float
ZE_F.argtypes = [Range]
VE_F = _libraries[librslPath].VE_F
VE_F.restype = c_float
VE_F.argtypes = [Range]
KD_F = _libraries[librslPath].KD_F
KD_F.restype = c_float
KD_F.argtypes = [Range]
TI_F = _libraries[librslPath].TI_F
TI_F.restype = c_float
TI_F.argtypes = [Range]
DX_F = _libraries[librslPath].DX_F
DX_F.restype = c_float
DX_F.argtypes = [Range]
CH_F = _libraries[librslPath].CH_F
CH_F.restype = c_float
CH_F.argtypes = [Range]
AH_F = _libraries[librslPath].AH_F
AH_F.restype = c_float
AH_F.argtypes = [Range]
CV_F = _libraries[librslPath].CV_F
CV_F.restype = c_float
CV_F.argtypes = [Range]
AV_F = _libraries[librslPath].AV_F
AV_F.restype = c_float
AV_F.argtypes = [Range]
SQ_F = _libraries[librslPath].SQ_F
SQ_F.restype = c_float
SQ_F.argtypes = [Range]
DZ_INVF = _libraries[librslPath].DZ_INVF
DZ_INVF.restype = Range
DZ_INVF.argtypes = [c_float]
VR_INVF = _libraries[librslPath].VR_INVF
VR_INVF.restype = Range
VR_INVF.argtypes = [c_float]
SW_INVF = _libraries[librslPath].SW_INVF
SW_INVF.restype = Range
SW_INVF.argtypes = [c_float]
CZ_INVF = _libraries[librslPath].CZ_INVF
CZ_INVF.restype = Range
CZ_INVF.argtypes = [c_float]
ZT_INVF = _libraries[librslPath].ZT_INVF
ZT_INVF.restype = Range
ZT_INVF.argtypes = [c_float]
DR_INVF = _libraries[librslPath].DR_INVF
DR_INVF.restype = Range
DR_INVF.argtypes = [c_float]
LR_INVF = _libraries[librslPath].LR_INVF
LR_INVF.restype = Range
LR_INVF.argtypes = [c_float]
ZD_INVF = _libraries[librslPath].ZD_INVF
ZD_INVF.restype = Range
ZD_INVF.argtypes = [c_float]
DM_INVF = _libraries[librslPath].DM_INVF
DM_INVF.restype = Range
DM_INVF.argtypes = [c_float]
RH_INVF = _libraries[librslPath].RH_INVF
RH_INVF.restype = Range
RH_INVF.argtypes = [c_float]
PH_INVF = _libraries[librslPath].PH_INVF
PH_INVF.restype = Range
PH_INVF.argtypes = [c_float]
XZ_INVF = _libraries[librslPath].XZ_INVF
XZ_INVF.restype = Range
XZ_INVF.argtypes = [c_float]
CD_INVF = _libraries[librslPath].CD_INVF
CD_INVF.restype = Range
CD_INVF.argtypes = [c_float]
MZ_INVF = _libraries[librslPath].MZ_INVF
MZ_INVF.restype = Range
MZ_INVF.argtypes = [c_float]
MD_INVF = _libraries[librslPath].MD_INVF
MD_INVF.restype = Range
MD_INVF.argtypes = [c_float]
ZE_INVF = _libraries[librslPath].ZE_INVF
ZE_INVF.restype = Range
ZE_INVF.argtypes = [c_float]
VE_INVF = _libraries[librslPath].VE_INVF
VE_INVF.restype = Range
VE_INVF.argtypes = [c_float]
KD_INVF = _libraries[librslPath].KD_INVF
KD_INVF.restype = Range
KD_INVF.argtypes = [c_float]
TI_INVF = _libraries[librslPath].TI_INVF
TI_INVF.restype = Range
TI_INVF.argtypes = [c_float]
DX_INVF = _libraries[librslPath].DX_INVF
DX_INVF.restype = Range
DX_INVF.argtypes = [c_float]
CH_INVF = _libraries[librslPath].CH_INVF
CH_INVF.restype = Range
CH_INVF.argtypes = [c_float]
AH_INVF = _libraries[librslPath].AH_INVF
AH_INVF.restype = Range
AH_INVF.argtypes = [c_float]
CV_INVF = _libraries[librslPath].CV_INVF
CV_INVF.restype = Range
CV_INVF.argtypes = [c_float]
AV_INVF = _libraries[librslPath].AV_INVF
AV_INVF.restype = Range
AV_INVF.argtypes = [c_float]
SQ_INVF = _libraries[librslPath].SQ_INVF
SQ_INVF.restype = Range
SQ_INVF.argtypes = [c_float]
radar_load_date_time = _libraries[librslPath].radar_load_date_time
radar_load_date_time.restype = None
radar_load_date_time.argtypes = [POINTER(Radar)]
big_endian = _libraries[librslPath].big_endian
big_endian.restype = c_int
big_endian.argtypes = []
little_endian = _libraries[librslPath].little_endian
little_endian.restype = c_int
little_endian.argtypes = []
swap_4_bytes = _libraries[librslPath].swap_4_bytes
swap_4_bytes.restype = None
swap_4_bytes.argtypes = [c_void_p]
swap_2_bytes = _libraries[librslPath].swap_2_bytes
swap_2_bytes.restype = None
swap_2_bytes.argtypes = [c_void_p]
hash_table_for_sweep = _libraries[librslPath].hash_table_for_sweep
hash_table_for_sweep.restype = POINTER(Hash_table)
hash_table_for_sweep.argtypes = [POINTER(Sweep)]
hash_bin = _libraries[librslPath].hash_bin
hash_bin.restype = c_int
hash_bin.argtypes = [POINTER(Hash_table), c_float]
the_closest_hash = _libraries[librslPath].the_closest_hash
the_closest_hash.restype = POINTER(Azimuth_hash)
the_closest_hash.argtypes = [POINTER(Azimuth_hash), c_float]
construct_sweep_hash_table = _libraries[librslPath].construct_sweep_hash_table
construct_sweep_hash_table.restype = POINTER(Hash_table)
construct_sweep_hash_table.argtypes = [POINTER(Sweep)]
angle_diff = _libraries[librslPath].angle_diff
angle_diff.restype = c_double
angle_diff.argtypes = [c_float, c_float]
rsl_query_field = _libraries[librslPath].rsl_query_field
rsl_query_field.restype = c_int
rsl_query_field.argtypes = [STRING]
pclose = _libraries[librslPath].pclose
pclose.restype = c_int
pclose.argtypes = [POINTER(FILE)]
__all__ = ['RSL_new_cappi', 'ctermid_r', 'getc_unlocked',
           'RAINBOW_FILE', 'fpos_t', 'RSL_volume_to_pict',
           'Histogram', 'RSL_get_linear_value', 'the_closest_hash',
           'TI_INVF', 'swblk_t',
           'RSL_get_window_from_volume', 'hash_table_for_sweep',
           'fclose', 'SQ_F', 'RSL_get_histogram_from_sweep',
           'RSL_write_radar_gzip', 'rsl_pclose', 'in_port_t',
           'RADTEC_FILE', 'swap_4_bytes', 'SQ_INVF', 'PH_F', 'fgetln',
           'RSL_select_fields', 'pthread_rwlock_t',
           'RSL_anyformat_to_radar', 'MZ_INVF', 'snprintf',
           'RSL_copy_volume', 'Cube_value',
           'RSL_get_first_sweep_of_volume',
           'RSL_get_window_from_radar', 'construct_sweep_hash_table',
           'RSL_free_histogram', 'RSL_get_window_from_sweep',
           'getchar_unlocked', 'RSL_bscan_ray', 'File_type',
           'RSL_sort_radar', 'RSL_get_groundr_and_h',
           'RSL_rhi_sweep_to_cart', 'Radar_header',
           'RSL_get_first_ray_of_volume', 'TOGA_FILE', 'NSIG_FILE_V2',
           'WSR88D_FILE', 'V_SW', '_pthread_handler_rec', 'popen',
           'pthread_t', 'Ray_header', 'RSL_load_vel_color_table',
           'RSL_rebin_velocity_ray', 'RSL_rebin_zdr_ray', '__P',
           'RSL_new_ray', 'RSL_lassen_to_radar',
           'RSL_read_these_sweeps', 'rlim_t', 'AV_INVF', 'getchar',
           'u_quad_t', '_opaque_pthread_rwlockattr_t',
           'RSL_get_histogram_from_volume', 'S_LR', 'MCGILL_FILE',
           'daddr_t', 'RSL_free_carpi', 'RSL_carpi_to_gif', 'minor',
           'RSL_sweep_z_to_r', 'KD_INVF', '_opaque_pthread_attr_t',
           'RSL_free_volume', 'RSL_rhi_sweep_to_gif', 'pid_t',
           'ftrylockfile', 'RSL_load_red_table', 'RSL_write_volume',
           'RSL_ray_z_to_r', 'pthread_key_t', 'FD_ZERO', 'u_int8_t',
           'RSL_write_sweep', 'RSL_bscan_volume', 'off_t',
           'RSL_add_dbz_offset_to_ray', 'fprintf', 'V_VR',
           'RSL_toga_to_radar', 'feof', 'clearerr',
           'RSL_add_dbz_offset_to_sweep', 'angle_diff',
           'RSL_load_color_table', 'rewind', 'putc_unlocked', 'SW_F',
           'flockfile', 'vasprintf', 'RSL_fill_cappi', 'key_t',
           'uint', 'RSL_find_rng_azm', 'pthread_rwlockattr_t',
           'funlockfile', '_opaque_pthread_t', 'Sweep', 'ssize_t',
           'RSL_load_rainfall_color_table',
           'RSL_fractional_area_of_sweep', 'u_long', '__srget',
           'RSL_rebin_zdr_sweep', 'RSL_rebin_zdr_volume', 'FILE',
           'ferror_unlocked', 'size_t', 'VE_F',
           'RSL_get_slantr_and_elev', 'LR_INVF', 'V_DZ',
           'RSL_new_radar', 'fwrite', 'RSL_radar_to_uf', 'RH_F',
           '__COPYRIGHT', 'V_DR', 'rsl_query_field', 'feof_unlocked',
           'CV_INVF', 'RSL_new_volume', 'TI_F',
           'RSL_load_sw_color_table', 'RSL_volume_to_cube',
           'RSL_write_pgm', 'u_char', 'fixpt_t', 'DX_INVF', 'uid_t',
           'u_int64_t', 'u_int16_t', 'RSL_get_color_table',
           'RSL_area_of_ray', 'Ray', 'vsscanf',
           'RSL_get_value_from_cappi', 'clock_t',
           'Slice_value', 'R_ZT', 'sprintf', 'vscanf',
           'RSL_prune_ray', 'u_int32_t', 'asprintf', 'ferror',
           'RSL_get_histogram_from_ray', 'RSL_write_radar',
           'RSL_radtec_to_radar', 'RSL_sweep_to_pict', 'fseeko',
           'putchar', 'RSL_get_ray_below', 'R_SW', 'pthread_attr_t',
           'XZ_F', 'ino_t', 'RSL_radar_verbose_on', 'major',
           'RSL_bscan_sweep', 'RSL_reverse_sweep_order', 'UNKNOWN',
           'little_endian', 'RSL_new_carpi', 
           'RSL_write_pict', 'RSL_volume_to_carpi',
           'RSL_radar_verbose_off', 'RSL_write_ray',
           'RSL_carpi_to_ppm', 'RSL_get_next_ccwise_ray', 'scanf',
           'ntohl', 'HDF_FILE', 'ntohs', 'htonl', 'RSL_hdf_to_radar',
           'RSL_sort_rays_in_sweep', 'setbuffer', 'RSL_carpi_to_cart',
           'RSL_fraction_of_sweep', 'RSL_read_histogram', 'HTONL',
           'fscanf', 'quad_t', 'CD_INVF', 'RSL_rebin_velocity_volume',
           'Volume_header', 'HTONS', 'vsnprintf', 'pthread_cond_t',
           'RSL_fraction_of_volume', 'Cappi',
           'RSL_sweep_to_cart', 'RSL_sweep_to_gif', 'nlink_t',
           'RSL_carpi_to_pgm', 'fmtcheck', 'RSL_fix_radar_header',
           'fseek', 'funopen', 'RSL_get_closest_sweep',
           'RSL_read_sweep', 'no_command', '__swbuf',
           'EDGE_FILE', 'KD_F', 'u_int', 'DM_INVF',
           'RSL_prune_volume', 'RSL_sweep_to_ppm', 'VE_INVF',
           '_opaque_pthread_rwlock_t', 'RSL_rapic_to_radar',
           'Hash_table', 'dev_t', 'setlinebuf', 'RSL_read_volume',
           'RSL_get_value_from_sweep', 'fpurge', 'RSL_volume_to_gif',
           'RSL_uf_to_radar', '_opaque_pthread_condattr_t',
           'putchar_unlocked', 'mode_t', 'Er_loc', 'fputc', 'qaddr_t',
           'fputs', 'ZD_INVF', 'RSL_uf_to_radar_fp', 'DZ_INVF',
           'intptr_t', 'RSL_get_next_cwise_ray', 'S_CZ',
           'RSL_fix_sweep_header', 'RSL_copy_ray', 'fileno', 'perror',
           'remove', 'fd_mask', 'AH_INVF', 'RSL_dorade_to_radar',
           'RSL_nsig2_to_radar', 'pthread_mutexattr_t',
           'RSL_get_ray_from_sweep', 'radar_load_date_time',
           'int16_t', 'DM_F', 'RSL_print_histogram', 'LASSEN_FILE',
           '__sbuf', 'RSL_sweep_to_pgm', 'fgetpos',
           'RSL_cappi_to_carpi', 'V_ZT', 'AH_F',
           'RSL_sort_rays_in_volume', 'V_LR', 'UF_FILE', '__RCSID',
           'ZE_INVF', 'RSL_free_radar', 'RSL_get_volume', 'V_CZ',
           '_opaque_pthread_cond_t', 'segsz_t', 'ushort', 'R_LR',
           'fd_set', 'caddr_t', 'ZD_F', 'int32_t',
           'RSL_nsig_to_radar', 'RSL_africa_to_radar', 'RSL_free_ray',
           'putw', 'puts', '_opaque_pthread_mutexattr_t',
           'clearerr_unlocked', 'RSL_new_sweep', 'RSL_EDGE_to_radar',
           'NSIG_FILE_V1', 'putc', 'RSL_mcgill_to_radar', 'CV_F',
           '_opaque_pthread_mutex_t', 'compress_pipe',
           'RSL_fix_volume_header', 'MD_F', 'RSL_get_ray',
           'RSL_get_closest_ray_from_sweep',
           'RSL_sort_sweeps_in_volume', 'Azimuth_hash',
           'RSL_fraction_of_ray', 'vsprintf', 'rename',
           'RSL_fix_time', 'Slice', 'DR_INVF', '__sFILE',
           'RSL_free_sweep', 'time_t', 'u_short', 'getc',
           '_azimuth_hash', 'RSL_clear_ray', 'ZE_F', 'big_endian',
           'gets', 'getw', '__sFILEX', 'DZ_F', 'setbuf',
           'RSL_get_ray_above', 'Cappi_value', 'in_addr_t',
           'RSL_sort_rays_by_time', 'RSL_volume_to_pgm',
           'pthread_mutex_t', 'RSL_get_window_from_ray',
           'RSL_prune_radar', 'RSL_z_to_r', 'RSL_volume_z_to_r',
           'RSL_get_nyquist_from_radar', 'pthread_condattr_t',
           'pthread_once_t', 'fflush', 'S_SW', 'register_t', 'CD_F',
           'DX_F', 'RSL_free_cube', 'RSL_load_height_color_table',
           'tmpnam', 'RSL_clear_volume', 'RSL_sort_volume',
           'RSL_get_first_ray_of_sweep', 'DR_F',
           'RSL_load_refl_color_table', 'R_VR', 'CH_F',
           'RSL_rebin_velocity_sweep', 'VR_F', 'RAPIC_FILE', 
           'DORADE_FILE', 'CZ_F', 'ZT_F', '__svfscanf',
           'RSL_clear_sweep', 'LR_F', 'freopen', 'tempnam', 'tmpfile',
           'RSL_print_version', 'fgetc', 'pclose', 'printf',
           'Sweep_header', 'RSL_write_gif', 'PH_INVF', 'fgets',
           'swap_2_bytes', 'howmany', 'ctermid',
           'RSL_allocate_histogram', 'Cube', 'fsetpos', 'ftell',
           'RSL_free_slice', 'R_CZ', 'RSL_get_sweep',
           'RSL_free_cappi', 'Carpi_value',
           'RSL_load_zdr_color_table', 'Range', 'R_DZ',
           'RSL_read_radar', 'AV_F', 'R_DR', 'MD_INVF', '__CONCAT',
           'S_VR', 'RSL_load_blue_table', 'RSL_get_slantr_and_h',
           'NTOHL', 'RSL_cappi_at_h', 'NTOHS', 'fopen',
           'RSL_set_color_table', 'RSL_prune_sweep', 'RSL_new_slice',
           'fdopen', 'Rsl_magic_num', 'S_DR', 'S_DZ', 'CZ_INVF',
           'uintptr_t', 'RSL_get_slice_from_cube', 'vprintf',
           'RSL_copy_sweep', 'RSL_carpi_to_pict', 'int8_t', 'ZT_INVF',
           'vfprintf', 'gid_t', 'RSL_new_cube',
           'RSL_radar_to_uf_gzip', 'RSL_FILE', 'Carpi', 'RH_INVF',
           'VR_INVF', 'SW_INVF', 'ungetc', 'MZ_F',
           'RSL_load_green_table', 'sscanf', 'RSL_rainbow_to_radar',
           'fread', 'ftello', '__STRING',
           'CH_INVF', 'int64_t', 'RSL_get_value_from_ray', 'htons',
           'hash_bin', 'RSL_wsr88d_to_radar', 'XZ_INVF',
           'RSL_get_value_at_h', 'RSL_add_dbz_offset_to_volume',
           'RSL_write_histogram', 'Volume', 'RSL_volume_to_ppm',
           'vfscanf', 'RSL_read_ray', 'RSL_get_value',
           'uncompress_pipe', 'useconds_t',
           'RSL_get_range_of_range_index', 'S_ZT', 'Radar', 'setvbuf',
           'RSL_write_ppm']

def getAllRays(radar, fieldType=None):
	"Return a list of all rays from a single field in the radar structure. Defaults to the reflectivity field"

	if fieldType is None:
		fieldType = fieldTypes().DZ
	
	allrays = []	
		
	swps = radar.contents.volumes[fieldType].sweeps
	for swp in swps:
		rays = swp.rays
		for ray in rays:
			allrays.append(ray)
	
	return allrays

vol_header_fields = dict(Volume_header._fields_)
f_prototype = vol_header_fields['f']
invf_prototype = vol_header_fields['invf']
# DZ_F_CFUNC = f_prototype(('DZ_F', _libraries[librslPath]))

# When creating a new radar structure, these need to be set for {vol,sweep,ray}.h.{f,invf}
conversion_functions = {
	'DZ' : (f_prototype(('DZ_F', _libraries[librslPath])), invf_prototype(('DZ_INVF', _libraries[librslPath]))),
	'VR' : (f_prototype(('VR_F', _libraries[librslPath])), invf_prototype(('VR_INVF', _libraries[librslPath]))),
	'SW' : (SW_F, SW_INVF),
	'CZ' : (CZ_F, CZ_INVF),
	'ZT' : (ZT_F, ZT_INVF),
	'DR' : (DR_F, DR_INVF),
	'LR' : (LR_F, LR_INVF),
	'ZD' : (ZD_F, ZD_INVF),
	'DM' : (DM_F, DM_INVF),
	'RH' : (RH_F, RH_INVF),
	'PH' : (PH_F, PH_INVF),
	'XZ' : (XZ_F, XZ_INVF),
	'CD' : (CD_F, CD_INVF),
	'MZ' : (MZ_F, MZ_INVF),
	'MD' : (MD_F, MD_INVF),
	'ZE' : (ZE_F, ZE_INVF),
	'VE' : (VE_F, VE_INVF),
	'KD' : (KD_F, KD_INVF),
	'TI' : (TI_F, TI_INVF),
	'DX' : (DX_F, DX_INVF),
	'CH' : (CH_F, CH_INVF),
	'AH' : (AH_F, AH_INVF),
	'CV' : (CV_F, CV_INVF),
	'AV' : (AV_F, AV_INVF),
	'SQ' : (SQ_F, SQ_INVF),
	}
