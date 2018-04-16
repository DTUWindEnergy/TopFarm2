from ctypes import cdll, c_char_p, c_double


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
                if isinstance(arg,float):
                    return c_double(arg)
                return arg
            err = func(*(map(fmt,args)))
            if err:
                raise Exception(err.decode())
            
        return func_wrap
