from urllib.request import urlopen, Request
from urllib.parse import urlencode
import time
from _collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def report_result(name, method=''):
    def wrap1(f):
        def wrap(*args, **kwargs):
            t = time.time()
            res = f(*args, **kwargs)
            ref = 365.8015799324624
            cost = -list(res)[0]
            duration = time.time() - t
            print("%s improved AEP to %f.4fGWh (%+.4f%%) in %.3fs" % (name, cost, (cost - ref) / ref * 100, duration))
            url = "http://tools.windenergy.dtu.dk/topfarm_ex/insert_result.asp"
            values = {'user': name,
                      'time': duration,
                      'aep': cost,
                      'comment': method,
                      }
            data = urlencode(values).encode("utf-8")
            req = Request(url, data)
            urlopen(req)
            return res
        return wrap
    return wrap1


def plot_result():
    url = "http://tools.windenergy.dtu.dk/topfarm_ex/view_result.asp"
    req = Request(url)
    s = urlopen(req).read().decode()
    res_dict = defaultdict(list)
    for r in s.split("<br>")[:-1]:
        name, duration, aep, comment = r.split(";")
        res_dict[name].append((duration, aep, comment))
    fig = plt.figure(figsize=(25, 10))
    ref_aep = 365.8015799324624
    duration_lst = []
    for i, name in enumerate(sorted(res_dict.keys())):
        duration = np.array([res[0] for res in res_dict[name]], dtype=float)
        duration_lst.extend(duration)
        aep = np.array([res[1] for res in res_dict[name]], dtype=float)
        marker = "v^<>.ospP*X"[i // 10]
        plt.plot(duration, (aep - ref_aep) / ref_aep * 100, '^', label=name, marker=marker)
    plt.plot([min(duration_lst), max(duration_lst)], [0, 0], 'gray')
    best = (418.9244064 - ref_aep) / ref_aep * 100
    plt.plot([min(duration_lst), max(duration_lst)], [best, best], 'gray')
    plt.xlabel('Time [s]')
    plt.ylabel('AEP improvement [%]')
    plt.legend()
    return fig


def reset_results():
    url = "http://tools.windenergy.dtu.dk/topfarm_ex/reset_results.asp"
    req = Request(url)
    urlopen(req)


if __name__ == '__main__':
    fig = plot_result()
    plt.show()
