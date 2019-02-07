from openmdao.recorders.case_recorder import CaseRecorder
from openmdao.core.driver import Driver
from openmdao.core.system import System
from openmdao.utils.record_util import values_to_array
from openmdao.recorders.case import Case
import numpy as np
import os
import pickle
import topfarm
from collections import OrderedDict
import re
from copy import deepcopy


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


class ListRecorder(CaseRecorder):
    def __init__(self, filepath=None, append=False, pickle_version=2, record_viewer_data=True):
        self.connection = None
        self._record_viewer_data = record_viewer_data
        self.format_version = 0
        self._record_on_proc = True
        self._database_initialized = False
        self._pickle_version = pickle_version
        self.filename = filepath
        self._filepath = filepath
        self.driver_iteration_dict = {}
        self.iteration_coordinate_lst = []
        self._abs2prom = {'input': {}, 'output': {}}
        self._prom2abs = {'input': {}, 'output': {}}
        self._abs2meta = {}
        self._driver_cases = None
        self.scaling_vecs = None
        self.user_options = None
        self.dtypes = []
        self.global_iterations = []
        self.problem_metadata = {'abs2prom': self._abs2prom,
                                 'connections_list': [],
                                 'tree': [],
                                 'variables': []}
        super(ListRecorder, self).__init__(record_viewer_data)

    def _initialize_database(self):
        pass

    def _cleanup_abs2meta(self):
        pass

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

    @property
    def num_cases(self):
        return len(self.iteration_coordinate_lst)

    @property
    def driver_cases(self):
        voi_meta = self.problem_metadata['variables']

        self._driver_cases = DriverCases(self._filepath, self.format_version,
                                         self.global_iterations,
                                         self._prom2abs, self._abs2prom,
                                         self._abs2meta, voi_meta,
                                         self.driver_iteration_dict,
                                         self.dtypes)
        self._driver_cases._case_keys = self.iteration_coordinate_lst
        self._driver_cases.num_cases = len(self._driver_cases._case_keys)
        return self._driver_cases

    def get(self, key):
        if isinstance(key, (tuple, list)):
            return np.array([self.get(k) for k in key]).T
        res = np.array(self.driver_iteration_dict[key])
        if len(res.shape) > 1 and res.shape[-1] == 1:
            res = np.squeeze(res, -1)
        return res

    def __getitem__(self, key):
        return self.get(key)

    def keys(self):
        return list(np.unique(['counter', 'iteration_coordinate', 'timestamp', 'success', 'msg'] +
                              list(self._prom2abs['input']) + list(self._prom2abs['output'])))

    def record_metadata_driver(self, recording_requester):
        pass

    def record_metadata_system(self, recording_requester):
        """
        Record system metadata.

        Parameters
        ----------
        recording_requester : System
            The System that would like to record its metadata.
        """
        self.scaling_vecs, self.user_options = self._get_metadata_system(recording_requester)

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
        input_, output = [values_to_array(data[n]) for n in ['in', 'out']]
        self.global_iterations.append(('driver', len(output), str(recording_requester)))
        meta_fields = [('counter', self._counter),
                       ('iteration_coordinate', self._iteration_coordinate),
                       ('timestamp', metadata['timestamp']),
                       ('success', metadata['success']),
                       ('msg', metadata['msg'])]

        data_fields = [(abs2prom[n], v)
                       for struct_arr, abs2prom in [(input_[0], self._abs2prom['input']), (output[0], self._abs2prom['output'])]
                       for n, v in zip(struct_arr.dtype.names, struct_arr)]

        # first time -> create lists
        if len(self.iteration_coordinate_lst) == 0:
            for k, _ in meta_fields + data_fields:
                self.driver_iteration_dict[k] = []
            self.dtypes = [input_.dtype, output.dtype]

        self.iteration_coordinate_lst.append(self._iteration_coordinate)
        for k, v in meta_fields:
            self.driver_iteration_dict[k].append(v)
        N = len(self.iteration_coordinate_lst)
        for k, v in data_fields:
            lst = self.driver_iteration_dict[k]
            if len(lst) < N:
                lst.append(v)
            else:
                #  # may be different due to scaling
                #  # we hope the first is the right one
                #  assert np.all(lst[-1] == v)
                pass

    def record_iteration_problem(self, recording_requester, data, metadata):
        pass

    def record_iteration_system(self, recording_requester, data, metadata):
        pass

    def record_iteration_solver(self, recording_requester, data, metadata):
        pass

    def record_viewer_data(self, model_viewer_data, key='Driver'):
        pass

    def record_metadata_solver(self, recording_requester):
        pass

    def record_derivatives_driver(self, recording_requester, data, metadata):
        pass

    def shutdown(self):
        pass

    def save(self, filename):
        d = {'driver_iteration_dict': self.driver_iteration_dict,
             'iteration_coordinate_lst': self.iteration_coordinate_lst,
             '_abs2prom': self._abs2prom,
             '_prom2abs': self._prom2abs,
             '_abs2meta': self._abs2meta,
             'dtypes': self.dtypes,
             'scaling_vecs': self.scaling_vecs,
             'user_options': self.user_options
             }

        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as fid:
            pickle.dump([self.__class__, d], fid)

    def load_if_exists(self, filename):
        if filename and os.path.isfile(filename):
            self.load(filename)
        return self

    def load(self, filename):
        if not filename or os.path.isfile(filename) is False:
            raise FileNotFoundError("No such file '%s'" % filename)
        with open(filename, 'rb') as fid:
            cls, attributes = pickle.load(fid)
        assert self.__class__ == cls
        self.__dict__.update(attributes)
        return self


# regular expression used to determine if a node in an iteration coordinate represents a system
_coord_system_re = re.compile('(_solve_nonlinear|_apply_nonlinear)$')

# Regular expression used for splitting iteration coordinates, removes separator and iter counts
_coord_split_re = re.compile('\|\\d+\|*')


def _get_source_system(iteration_coordinate):
    """
    Get pathname of system that is the source of the iteration.

    Parameters
    ----------
    iteration_coordinate : str
        The full unique identifier for this iteration.

    Returns
    -------
    str
        The pathname of the system that is the source of the iteration.
    """
    path = []
    parts = _coord_split_re.split(iteration_coordinate)
    for part in parts:
        if (_coord_system_re.search(part) is not None):
            if ':' in part:
                # get rid of 'rank#:'
                part = part.split(':')[1]
            path.append(part.split('.')[0])

    # return pathname of the system
    return '.'.join(path)


class CaseTable(object):
    """
    Base class for wrapping case tables in a recording database.

    Attributes
    ----------
    _filename : str
        The name of the recording file from which to instantiate the case reader.
    _format_version : int
        The version of the format assumed when loading the file.
    _table_name : str
        The name of the table in the database.
    _index_name : str
        The name of the case index column in the table.
    _global_iterations : list
        List of iteration cases and the table and row in which they are found.
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _abs2meta : dict
        Dictionary mapping absolute variable names to variable metadata.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _voi_meta : dict
        Dictionary mapping absolute variable names to variable settings.
    _sources : list
        List of sources of cases in the table.
    _keys : list
        List of keys of cases in the table.
    _cases : dict
        Dictionary mapping keys to cases that have already been loaded.
    _global_iterations : list
        List of iteration cases and the table and row in which they are found.
    """

    def __init__(self, fname, ver, table, index, giter, prom2abs, abs2prom, abs2meta, voi_meta):
        """
        Initialize.

        Parameters
        ----------
        fname : str
            The name of the recording file from which to instantiate the case reader.
        ver : int
            The version of the format assumed when loading the file.
        table : str
            The name of the table in the database.
        index : str
            The name of the case index column in the table.
        giter : list of tuple
            The global iterations table.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        abs2meta : dict
            Dictionary mapping absolute variable names to variable metadata.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        voi_meta : dict
            Dictionary mapping absolute variable names to variable settings.
        """
        self._filename = fname
        self._format_version = ver
        self._table_name = table
        self._index_name = index
        self._global_iterations = giter
        self._prom2abs = prom2abs
        self._abs2prom = abs2prom
        self._abs2meta = abs2meta
        self._voi_meta = voi_meta

        # cached keys/cases
        self._sources = None
        self._keys = None
        self._cases = {}

    def count(self):
        """
        Get the number of cases recorded in the table.

        Returns
        -------
        int
            The number of cases recorded in the table.
        """
        pass

    def list_cases(self, source=None):
        """
        Get list of case IDs for cases in the table.

        Parameters
        ----------
        source : str, optional
            A source of cases or the iteration coordinate of a case.
            If not None, only cases originating from the specified source or case are returned.

        Returns
        -------
        list
            The cases from the table from the specified source or parent case.
        """
        pass

    def get_cases(self, source=None, recurse=False, flat=False):
        """
        Get list of case names for cases in the table.

        Parameters
        ----------
        source : str, optional
            If not None, only cases that have the specified source will be returned
        recurse : bool, optional
            If True, will enable iterating over all successors in case hierarchy
        flat : bool, optional
            If False and there are child cases, then a nested ordered dictionary
            is returned rather than an iterator.

        Returns
        -------
        list or dict
            The cases from the table that have the specified source.
        """
        pass

    def get_case(self, case_id, cache=False):
        """
        Get a case from the database.

        Parameters
        ----------
        case_id : str or int
            The string-identifier of the case to be retrieved or the index of the case.
        cache : bool
            If True, case will be cached for faster access by key.

        Returns
        -------
        Case
            The specified case from the table.
        """
        pass

    def _get_iteration_coordinate(self, case_idx):
        """
        Return the iteration coordinate for the indexed case (handles negative indices, etc.).

        Parameters
        ----------
        case_idx : int
            The case number that we want the iteration coordinate for.

        Returns
        -------
        iteration_coordinate : str
            The iteration coordinate.
        """
        pass

    def cases(self, cache=False):
        """
        Iterate over all cases, optionally caching them into memory.

        Parameters
        ----------
        cache : bool
            If True, cases will be cached for faster access by key.
        """
        pass

    def _load_cases(self):
        """
        Load all cases into memory.
        """
        pass

    def list_sources(self):
        """
        Get the list of sources that recorded data in this table.

        Returns
        -------
        list
            List of sources.
        """
        pass

    def _get_source(self, iteration_coordinate):
        """
        Get the source of the iteration.

        Parameters
        ----------
        iteration_coordinate : str
            The full unique identifier for this iteration.

        Returns
        -------
        str
            The source of the iteration.
        """
        return _get_source_system(iteration_coordinate)

    def _get_row_source(self, row_id):
        """
        Get the source of the case at the specified row of this table.

        Parameters
        ----------
        row_id : int
            The row_id of the case in the table.

        Returns
        -------
        str
            The source of the case.
        """
        pass

    def _get_first(self, source):
        """
        Get the first case from the specified source.

        Parameters
        ----------
        source : str
            The source.

        Returns
        -------
        Case
            The first case from the specified source.
        """
        pass


class DriverCases(CaseTable):
    """
    Case specific to the entries that might be recorded in a Driver iteration.
    """

    def __init__(self, filename, format_version, giter, prom2abs, abs2prom, abs2meta, voi_meta, driver_iteration_dict, dtypes):
        self.dtypes = dtypes
        super(DriverCases, self).__init__(filename, format_version,
                                          'driver_iterations', 'iteration_coordinate', giter,
                                          prom2abs, abs2prom, abs2meta, voi_meta)
        self._voi_meta = voi_meta
        self.driver_iteration_dict = driver_iteration_dict
        self.meta_field_names = ['counter', 'iteration_coordinate', 'timestamp', 'success', 'msg']
        self.abs2proms = [self._abs2prom[io] for io in ['input', 'output']]

    def cases(self, cache=False):
        pass

    def get_case(self, case_id):
        """
        Get a case from the database.

        Parameters
        ----------
        case_id : int or str
            The integer index or string-identifier of the case to be retrieved.

        Returns
        -------
            An instance of a Driver Case populated with data from the
            specified case/iteration.
        """
        did = self.driver_iteration_dict
        data = {}
        for key in did:
            data[key] = did[key][case_id]
#
        inputs, outputs = [np.array([tuple([did[abs2prom[k]][case_id] for k in dtype.names])], dtype=dtype)
                           for abs2prom, dtype in zip(self.abs2proms, self.dtypes)]
        data['inputs'] = inputs
        data['outputs'] = outputs
        case = Case('driver', data,
                    self._prom2abs, self._abs2prom, self._abs2meta, self._voi_meta,
                    self._format_version)

        return case


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


class TopFarmListRecorder(ListRecorder):

    def __init__(self, record_id=None):
        filepath, _ = recordid2filename(record_id)
        ListRecorder.__init__(self, filepath)
        self.load_if_exists(record_id)

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
        ListRecorder.save(self, filename)

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

        ListRecorder.load(self, filename)
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
