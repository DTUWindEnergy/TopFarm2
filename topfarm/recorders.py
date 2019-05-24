from openmdao.api import SqliteRecorder
import numpy as np
import os
import pickle
import topfarm
from copy import deepcopy
from openmdao.core.driver import Driver
from openmdao.core.system import System


def split_record_id(record_id):
    if record_id is None:
        return "", ""
    folder, record_id = os.path.split(record_id)
    filename, load_case, *_ = (record_id + ":latest").split(":")
    if load_case == "":
        load_case = 'latest'
    return os.path.join(folder, filename).replace("\\", "/"), load_case


def recordid2filename(record_id):
    if record_id is None:
        return "", ""
    filename, load_case = split_record_id(record_id)

    if not filename.lower().endswith(".pkl"):
        filename += ".pkl"
    folder, filename = os.path.split(filename)
    if folder == "":
        folder = "recordings"
    return os.path.join(folder, filename).replace("\\", "/"), load_case.lower()


def convert_to_list(vals):
    """
    Recursively convert arrays, tuples, and sets to lists.

    Parameters
    ----------
    vals : numpy.array or list or tuple
        the object to be converted to a list

    Returns
    -------
    list :
        The converted list.
    """
    if isinstance(vals, np.ndarray):
        return convert_to_list(vals.tolist())
    elif isinstance(vals, (list, tuple, set)):
        return [convert_to_list(item) for item in vals]
    else:
        return vals


class TopFarmListRecorder(SqliteRecorder):
    def __init__(self, record_id=None, filepath='cases.sql', append=False, pickle_version=2, record_viewer_data=False):
        super().__init__(filepath, append, pickle_version, record_viewer_data)
        self.iteration_coordinate_lst = []
        self.filepath = filepath
        self.driver_iteration_dict = {}
        filepath, _ = recordid2filename(record_id)
        self.load_if_exists(record_id)
        self.meta_field_names = ['counter', 'iteration_coordinate', 'timestamp', 'success', 'msg']
        self._abs2prom = {'input': {}, 'output': {}}
        self._prom2abs = {'input': {}, 'output': {}}
        self._abs2meta = {}

    def startup(self, recording_requester):
        """
        Prepare for a new run and create/update the abs2prom and prom2abs variables.

        Parameters
        ----------
        recording_requester : object
            Object to which this recorder is attached.
        """
        super().startup(recording_requester)

        # grab the system
        if isinstance(recording_requester, Driver):
            system = recording_requester._problem.model
        elif isinstance(recording_requester, System):
            system = recording_requester
        else:
            system = recording_requester._system

        # grab all of the units and type (collective calls)
        states = system._list_states_allprocs()
        desvars = system.get_design_vars(True)
        responses = system.get_responses(True)
        objectives = system.get_objectives(True)
        constraints = system.get_constraints(True)
        inputs = system._var_allprocs_abs_names['input']
        outputs = system._var_allprocs_abs_names['output']
        full_var_set = [(inputs, 'input'), (outputs, 'output'),
                        (desvars, 'desvar'), (responses, 'response'),
                        (objectives, 'objective'), (constraints, 'constraint')]

        # merge current abs2prom and prom2abs with this system's version
        for io in ['input', 'output']:
            for v in system._var_abs2prom[io]:
                self._abs2prom[io][v] = system._var_abs2prom[io][v]
            for v in system._var_allprocs_prom2abs_list[io]:
                if v not in self._prom2abs[io]:
                    self._prom2abs[io][v] = system._var_allprocs_prom2abs_list[io][v]
                else:
                    self._prom2abs[io][v] = list(set(self._prom2abs[io][v]) |
                                                 set(system._var_allprocs_prom2abs_list[io][v]))

        for var_set, var_type in full_var_set:
            for name in var_set:
                if name not in self._abs2meta:
                    self._abs2meta[name] = system._var_allprocs_abs2meta[name].copy()
                    self._abs2meta[name]['type'] = set()
                    if name in states:
                        self._abs2meta[name]['explicit'] = False

                if var_type not in self._abs2meta[name]['type']:
                    self._abs2meta[name]['type'].add(var_type)
                self._abs2meta[name]['explicit'] = True

        for name in inputs:
            self._abs2meta[name] = system._var_allprocs_abs2meta[name].copy()
            self._abs2meta[name]['type'] = set()
            self._abs2meta[name]['type'].add('input')
            self._abs2meta[name]['explicit'] = True
            if name in states:
                self._abs2meta[name]['explicit'] = False

        var_settings = {}
        var_settings.update(desvars)
        var_settings.update(objectives)
        var_settings.update(constraints)
        var_settings = self._cleanup_var_settings(var_settings)

    def _cleanup_var_settings(self, var_settings):
        """
        Convert all var_settings variable properties to a form that can be dumped as JSON.

        Parameters
        ----------
        var_settings : dict
            Dictionary mapping absolute variable names to variable settings.

        Returns
        -------
        var_settings : dict
            Dictionary mapping absolute variable names to var settings that are JSON compatible.
        """
        # otherwise we trample on values that are used elsewhere
        var_settings = deepcopy(var_settings)
        for name in var_settings:
            for prop in var_settings[name]:
                val = var_settings[name][prop]
                if isinstance(val, np.int8) or isinstance(val, np.int16) or\
                   isinstance(val, np.int32) or isinstance(val, np.int64):
                    var_settings[name][prop] = val.item()
                elif isinstance(val, tuple):
                    var_settings[name][prop] = [int(v) for v in val]
                elif isinstance(val, np.ndarray):
                    var_settings[name][prop] = convert_to_list(var_settings[name][prop])

        return var_settings

    def get(self, key):
        if isinstance(key, (tuple, list)):
            return np.array([self.get(k) for k in key]).T
        res = np.array(self.driver_iteration_dict[key])
        if len(res.shape) > 1 and res.shape[-1] == 1:
            res = np.squeeze(res, -1)
        return res

    def __getitem__(self, key):
        return self.get(key)

    @property
    def num_cases(self):
        return len(self.iteration_coordinate_lst)

    def record_iteration_driver(self, recording_requester, data, metadata):
        """
        Record data and metadata from a Driver.

        Parameters
        ----------
        recording_requester : object
            Driver in need of recording.
        data : dict
            Dictionary containing desvars, objectives, constraints, responses, and System vars.
        metadata : dict
            Dictionary containing execution metadata.
        """
        self.iteration_coordinate_lst.append(self._iteration_coordinate)
        meta_fields = [('counter', self._counter),
                       ('iteration_coordinate', self._iteration_coordinate),
                       ('timestamp', metadata['timestamp']),
                       ('success', metadata['success']),
                       ('msg', metadata['msg'])]

        out_keys = []
        in_keys = []
        if self.num_cases == 1:
            for key in data['out']:
                rec_key = key.split('.')[-1]
                out_keys.append(rec_key)
                self.driver_iteration_dict[rec_key] = [data['out'][key]]
            for key in data['in']:
                rec_key = key.split('.')[-1]
                if rec_key not in out_keys:
                    self.driver_iteration_dict[rec_key] = [data['in'][key]]
            for k, v in meta_fields:
                self.driver_iteration_dict[k] = [v]

        else:
            for key in data['out']:
                rec_key = key.split('.')[-1]
                self.driver_iteration_dict[rec_key].append(data['out'][key])
            for key in data['in']:
                rec_key = key.split('.')[-1]
                if rec_key in in_keys:
                    self.driver_iteration_dict[rec_key].append(data['in'][key])
            for k, v in meta_fields:
                self.driver_iteration_dict[k].append(v)

    def _initialize_database(self):
        pass

    def animate_turbineXY(self, duration=10, tail=5, plot_initial=True, filename=None):
        import matplotlib.pyplot as plt
        x, y = self[topfarm.x_key], self[topfarm.y_key]
        cost = self.get('cost')
        boundary = self['xy_boundary'][-1]
        boundary = np.r_[boundary, boundary[:1]]
        n_wt = x.shape[1]
        from matplotlib import animation

        # color_cycle = iter(matplotlib.rcParams['axes.prop_cycle'])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        colors = colors * (n_wt // 10 + 1)

        fig, ax = plt.subplots()
        ln = [plt.plot([], [], '.-', color=c, animated=True)[0] for c in colors]
        ln += [plt.plot([], [], '--', color=c, animated=True)[0] for c in colors]
        ln += [plt.plot([], [], 'xk', animated=True)[0] for _ in colors]
        title = ax.text(0.5, .95, "", bbox={'facecolor': 'w', 'alpha': .8, 'pad': 5},
                        transform=ax.transAxes, ha="center")
        ln += title,

        def init():
            if len(boundary) > 0:
                plt.plot(boundary[:, 0], boundary[:, 1], 'k')
            else:
                ax.set_xlim([np.min(x), np.max(x)])
                ax.set_ylim([np.min(y), np.max(y)])
            plt.axis('equal')
            return ln

        init()

        def update(frame):
            title.set_text("%f (%.2f%%)" % (cost[frame],
                                            (cost[frame] - cost[0]) / cost[0] * 100))
            for i in range(n_wt):
                ln[i].set_data(x[max(0, frame - tail):frame + 1, i], y[max(0, frame - tail):frame + 1, i])
                if plot_initial:
                    ln[i + n_wt].set_data(np.r_[x[0, i], x[frame, i]], np.r_[y[0, i], y[frame, i]])
                ln[i + 2 * n_wt].set_data(x[frame, i], y[frame, i])

            return ln

        ani = animation.FuncAnimation(fig, update, frames=len(x),
                                      init_func=init, blit=True, interval=duration / len(x) * 1000, repeat=False)
        if filename:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=len(x) / duration, metadata=dict(artist='Me'), bitrate=1800)
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            ani.save(filename, writer=writer)
        else:
            plt.show()

    def split_record_id(self, record_id):
        return split_record_id(record_id)

    def recordid2filename(self, record_id):
        return recordid2filename(record_id)

    def save(self, record_id=None):
        record_id = record_id or self.filename
        filename, _ = self.recordid2filename(record_id)
        self.list_save(filename)

    def list_save(self, filename):
        d = {'driver_iteration_dict': self.driver_iteration_dict,
             'iteration_coordinate_lst': self.iteration_coordinate_lst,
             '_abs2prom': self._abs2prom,
             '_prom2abs': self._prom2abs,
             '_abs2meta': self._abs2meta,
             }

        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as fid:
            pickle.dump([self.__class__, d], fid)

    def load_if_exists(self, record_id):
        self.filename = self.recordid2filename(record_id)[0]
        if self.filename and os.path.isfile(self.filename):
            self.load(record_id)
        return self

    def load(self, record_id):
        filename, load_case = self.recordid2filename(record_id)
        if not filename or os.path.isfile(filename) is False:
            raise FileNotFoundError("No such file '%s'" % filename)

        if load_case == 'none' or load_case == '0':
            return self

        self.list_load(filename)
        if load_case == 'latest':
            load_case = None
        elif load_case == 'best':
            load_case = np.argmin(self.get('cost')) + 1
        else:
            load_case = int(load_case)
        self.driver_iteration_dict = {k: v[:load_case] for k, v in self.driver_iteration_dict.items()}
        self.iteration_coordinate_lst = self.iteration_coordinate_lst[:load_case]

        self.filename, self.load_case = filename, load_case
        return self

    def list_load(self, filename):
        if not filename or os.path.isfile(filename) is False:
            raise FileNotFoundError("No such file '%s'" % filename)
        with open(filename, 'rb') as fid:
            cls, attributes = pickle.load(fid)
        assert self.__class__ == cls
        self.__dict__.update(attributes)
        return self

    def keys(self):
        return list(np.unique(['counter', 'iteration_coordinate', 'timestamp', 'success', 'msg'] +
                              list(self._prom2abs['input']) + list(self._prom2abs['output'])))


class NestedTopFarmListRecorder(TopFarmListRecorder):
    def __init__(self, nested_comp, record_id=None):
        TopFarmListRecorder.__init__(self, record_id)
        self.nested_comp = nested_comp

    def record_iteration_driver(self, recording_requester, data, metadata):
        """
        Record data and metadata from a Driver.

        Parameters
        ----------
        recording_requester : object
            Driver in need of recording.
        data : dict
            Dictionary containing desvars, objectives, constraints, responses, and System vars.
        metadata : dict
            Dictionary containing execution metadata.
        """
        recorder = getattr(self.nested_comp.problem, 'recorder', None)
        if self.num_cases == 0:
            self.driver_iteration_dict['recorder'] = []
        TopFarmListRecorder.record_iteration_driver(self, recording_requester, data, metadata)

        self.driver_iteration_dict['recorder'].append(recorder)
