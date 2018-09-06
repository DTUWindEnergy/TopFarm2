from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.recorders.cases import BaseCases
from openmdao.recorders.case import PromotedToAbsoluteMap, DriverCase
from openmdao.recorders.base_case_reader import BaseCaseReader
from openmdao.core.driver import Driver
from openmdao.core.system import System
from openmdao.utils.record_util import values_to_array
import numpy as np
import matplotlib
import os
import pickle


class ListRecorder(BaseRecorder):
    def __init__(self):
        BaseRecorder.__init__(self)
        self.driver_iteration_lst = []
        self.iteration_coordinate_lst = []
        self._abs2prom = {'input': {}, 'output': {}}
        self._prom2abs = {'input': {}, 'output': {}}
        self._abs2meta = {}
        self._driver_cases = None
        self.scaling_vecs = None
        self.user_options = None

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
                
    @property
    def num_cases(self):
        return len(self.driver_iteration_lst)

    @property
    def driver_cases(self):
        self._driver_cases = DriverCases(self.driver_iteration_lst, self._abs2prom,
                                         self._abs2meta, self._prom2abs)
        self._driver_cases._case_keys = self.iteration_coordinate_lst
        self._driver_cases.num_cases = len(self._driver_cases._case_keys)
        self._driver_metadata = self.model_viewer_data
        return self._driver_cases

    def get(self, key):
        if len(self.driver_iteration_lst) == 0:
            raise ValueError("Driver iteration list empty")
        if isinstance(key, (tuple, list)):
            return np.array([self.get(k) for k in key]).T
        meta = ['counter', 'iteration_coordinate', 'timestamp', 'success', 'msg']
        if key in meta:
            i = meta.index(key)
            return np.array([r[i] for r in self.driver_iteration_lst])
        elif key in self._prom2abs['input']:
            abs_name = self._prom2abs['input'][key][0]
            i = self.driver_iteration_lst[0][5].dtype.names.index(abs_name)
            res = np.array([r[5][0][i] for r in self.driver_iteration_lst])
        elif key in self._prom2abs['output']:
            abs_name = self._prom2abs['output'][key][0]
            i = self.driver_iteration_lst[0][6].dtype.names.index(abs_name)
            res = np.array([r[6][0][i] for r in self.driver_iteration_lst])
        else:
            raise KeyError("'%s' not found in meta, input or output" % key)
        if res.shape[-1] == 1:
            res = res[:, 0]
        return res
    
    def __getitem__(self, key):
        return self.get(key)

    def keys(self):
        return list(np.unique(['counter', 'iteration_coordinate', 'timestamp', 'success', 'msg'] +
                              list(self._prom2abs['input']) + list(self._prom2abs['output'])))

    def record_metadata_driver(self, recording_requester):
        self.driver_class = type(recording_requester).__name__
        self.model_viewer_data = recording_requester._model_viewer_data

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
        self.iteration_coordinate_lst.append(self._iteration_coordinate)
        self.driver_iteration_lst.append(
            (self._counter, self._iteration_coordinate,
             metadata['timestamp'], metadata['success'], metadata['msg'],
             input_, output))

    def save(self, filename):

        d = {'driver_iteration_lst': [r[:7] for r in self.driver_iteration_lst],
             'iteration_coordinate_lst': self.iteration_coordinate_lst,
             '_abs2prom': self._abs2prom,
             '_prom2abs': self._prom2abs,
             '_abs2meta': self._abs2meta,
             'driver_class': self.driver_class,
             'model_viewer_data': self.model_viewer_data,
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
        if not filename or os.path.isfile(filename) == False:
            raise FileNotFoundError("No such file '%s'" % filename)
        with open(filename, 'rb') as fid:
            cls, attributes = pickle.load(fid)
        assert self.__class__ == cls
        self.__dict__.update(attributes)
        return self


class DriverCases(BaseCases):
    """
    Case specific to the entries that might be recorded in a Driver iteration.
    """

    def __init__(self, driver_iteration_lst, abs2prom, abs2meta, prom2abs):
        BaseCases.__init__(self, "None", abs2prom, abs2meta, prom2abs)
        self.driver_iteration_lst = driver_iteration_lst

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

        counter, iteration_coordinate, timestamp, success, msg, inputs, \
            outputs, *_ = self.driver_iteration_lst[case_id]

        case = DriverCase(self.filename, counter, iteration_coordinate, timestamp, success, msg,
                          inputs, outputs,
                          self._prom2abs, self._abs2prom, self._abs2meta)

        return case


class TopFarmListRecorder(ListRecorder):

    def __init__(self, record_id=None):
        ListRecorder.__init__(self)
        self.load_if_exists(record_id)

    def animate_turbineXY(self, duration=10, tail=5, plot_initial=True, filename=None):
        import matplotlib.pyplot as plt
        # plt.plot()
        x, y = self.get('turbineX'), self.get('turbineY')
        cost = self.get('cost')
        boundary = self.get('boundary')[0]
        boundary = np.r_[boundary, boundary[:1]]
#         plt.plot(boundary[:,0], boundary[:,1],'k')
#         plt.plot(x[:,0],y[:,0],'.-')
#         plt.plot(x[:,1],y[:,1],'.-')
#         plt.show()
        n_wt = x.shape[1]
        import matplotlib.pyplot as plt
        from matplotlib import animation

        color_cycle = iter(matplotlib.rcParams['axes.prop_cycle'])
        colors = [next(color_cycle)['color'] for _ in range(n_wt)]

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

    def recordid2filename(self, record_id):
        if record_id is None:
            return "", ""
        folder, record_id = os.path.split(record_id)
        filename, load_case, *_ = (record_id + ":latest").split(":")
        if load_case == "":
            load_case = 'latest'
        if not filename.lower().endswith(".pkl"):
            filename += ".pkl"
        if folder == "":
            folder = "recordings"
        return os.path.join(folder, filename).replace("\\", "/"), load_case.lower()

    def save(self, record_id=None):
        record_id = record_id or self.filename
        filename, _ = self.recordid2filename(record_id)
        ListRecorder.save(self, filename)

    def load_if_exists(self, record_id):
        self.filename, _ = self.recordid2filename(record_id)
        if self.filename and os.path.isfile(self.filename):
            self.load(record_id)
        return self

    def load(self, record_id):
        filename, load_case = self.recordid2filename(record_id)

        ListRecorder.load(self, filename)
        if load_case == 'latest':
            load_case = None
        elif load_case == 'best':
            load_case = np.argmin(self.get('cost')) + 1
        elif load_case == 'none':
            load_case = 0
        else:
            load_case = int(load_case)
        self.driver_iteration_lst = self.driver_iteration_lst[:load_case]
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
        TopFarmListRecorder.record_iteration_driver(self, recording_requester, data, metadata)
        self.driver_iteration_lst[-1] = self.driver_iteration_lst[-1] + (recorder,)

    def get(self, key):
        if key == 'recorder':
            return [r[-1] for r in self.driver_iteration_lst]
        return ListRecorder.get(self, key)

    def keys(self):
        return TopFarmListRecorder.keys(self) + ['recorder']
