from contextlib import contextmanager
from ctypes import cdll, c_char_p, c_double
import ctypes
import io
import os
import sys
import tempfile


from contextlib import contextmanager
import ctypes
import io
import os
import sys
import tempfile

import os
import sys
from contextlib import contextmanager


@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        #         if sys.stdout is not sys.__stdout__:
        #             sys.stdout.close() # + implicit flush()

        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        # sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        _redirect_stdout(to)
        # with open(to, 'w') as file:
        #    _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


class PascalDLL(object):
    def __init__(self, path):
        self.lib = cdll.LoadLibrary(path)
        
    def __getattr__(self, name):
        def func_wrap(*args):
            func = getattr(self.lib, name)
            func.restype = c_char_p

            def fmt(arg):
                if isinstance(arg, str):
                    return arg.encode()
                if isinstance(arg, float):
                    return c_double(arg)
                return arg
            
            fmt_args = list(map(fmt, args))
            err = func(*fmt_args)
            if err:
                raise Exception("Exception from Fugalib: " + err.decode())


#             with tempfile.TemporaryFile(mode='w+') as fp:
# #                 with stdout_redirected(to=fp):
# #                     fmt_args = map(fmt, args)
# #                     err = func(*fmt_args)
#                 fmt_args = map(fmt, args)
#                 err = func(*fmt_args)
# #                 fp.seek(0)
# #                 self.stdout += fp.read()
#                 if err:
#                     #print ("#"+self.stdout+"#")
#                     raise Exception("Exception from Fugalib: " + err.decode())
        return func_wrap
