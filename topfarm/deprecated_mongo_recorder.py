"""
Class definition for MongoRecorder, which provides dictionary inspired by SQLite.
"""

from copy import deepcopy
from collections import OrderedDict
import os
from itertools import chain
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
# from six import iteritems
from openmdao.recorders.case_recorder import CaseRecorder
from openmdao.core.driver import Driver
from openmdao.core.system import System
from openmdao.core.problem import Problem
from openmdao.solvers.solver import Solver
from pymongo import MongoClient
from topfarm.drivers.random_search_driver import RandomSearchDriver
import random
from datetime import datetime
# import pandas as pd
# separator, cannot be a legal char for names
META_KEY_SEP = '!'

_container_classes = (list, tuple, set)


def make_serializable(o):
    """
    Recursively convert numpy types to native types for JSON serialization.
    Parameters
    ----------
    o : object
        the object to be converted
    Returns
    -------
    object
        The converted object.
    """
    if isinstance(o, _container_classes):
        return [make_serializable(item) for item in o]
    elif isinstance(o, np.number):
        return o.item()
    elif isinstance(o, np.ndarray):
        return make_serializable(o.tolist())
    elif hasattr(o, '__dict__'):
        return make_serializable(o.__class__.__name__)
    else:
        return clean_keys(o)


def clean_str(k):
    """
    Function to replace '.' with '%'.
    Used when writing to Mongo database.
    Helper function for clean_keys() function.

    Parameters
    ----------
    k : str

    Returns
    -------
    updated k with replaced "." for "%"
    """
    return k.replace('.', '%')


def clean_keys(o):
    """
    Function removes extracts keys from layered dict.
    Helper function for make_serializable() function.

    Parameters
    ----------
    o : object
        If objects is a dict, flattens.

    Returns
    -------
    Updated o : object
        Flattened dict where applicable
    """
    if isinstance(o, dict):
        new_o = {}
        for k, v in o.items():
            new_o[clean_str(k)] = v
        return new_o
    else:
        return o


class RemoteMongo():
    def __init__(self, uri=None, host=None, port=27017, ip='localhost', password=None, user=None, db_name='database', uri_type=None, **kwargs):
        self.host = host
        self.port = port
        self.ip = ip
        self.uri = uri
        self.user = user
        self.password = password
        self.db_name = db_name
        self.uri_type = uri_type
        self.kwargs = kwargs

    def open(self):
        if self.uri:
            uri = self.uri
            self.kwargs.update({'host': uri})
        elif self.host:
            if not self.uri_type:
                uri = f"mongodb://{self.user}:{self.password}@{self.host}"
            elif self.uri_type == 'srv':
                uri = f"mongodb+srv://{self.user}:{self.password}@{self.host}"
        else:
            self.kwargs.update({'host': f'mongodb://{self.ip}:{self.port}',
                                'username': self.user,
                                'password': self.password,
                                'authSource': 'database_str'})

        self.client = MongoClient(**self.kwargs)
        self.db = self.client[self.db_name]
        return self.client

    def __enter__(self):
        self.open()
        return self.db

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.client.close()


class MongoRecorder(CaseRecorder):
    """
    Recorder that saves cases in Mongo DB.

    Attributes
    ----------
    _record_viewer_data : bool
        Flag indicating whether to record data needed to generate N2 diagram.
    connection :  connection object
        Connection to the database.
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _abs2meta : {'name': {}}
        Dictionary mapping absolute variable names to their metadata including units,
        bounds, and scaling.
    _database_initialized : bool
        Flag indicating whether or not the database has been initialized.
    _record_on_proc : bool
        Flag indicating whether to record on this processor when running in parallel.
    """

    def __init__(self, uri=None, host=None, ip='localhost', port=27017, user=None, password=None, uri_type=None, record_viewer_data=True, ssh_tunnel=False,
                 db_name='database', case_id=None, clean_up=False, with_mpi=False, client_args={}):
        """
        Initialize the MongoRecorder.

        Parameters
        ----------
        ip: str
            ip of the mongo server
        port: int
            port of the mongo server
        record_viewer_data : bool, optional
            If True, record data needed for visualization.
        """
        self._ip = ip
        self._port = port
        self._host = host
        self._uri = uri
        self._user = user
        self._password = password
        self._uri_type = uri_type
        self._client_args = client_args
        self.case_id = case_id
        self.db_name = db_name
        self.ssh_tunnel = ssh_tunnel
        self._record_viewer_data = record_viewer_data
        self._record_metadata = True
        self.metadata_connection = None
        self._started = set()

        self._abs2prom = {'input': {}, 'output': {}}
        self._prom2abs = {'input': {}, 'output': {}}
        self._abs2meta = {}
        self._database_initialized = False

        # default to record on all procs when running in parallel
        self._record_on_proc = True
        self.lastrowid = 0

        self.connection = True

        # testing
        self.driver_iteration_dict = {}

        # Generating a random id
        import time
        random.seed(int(time.time()))
        self.run_id = int(random.random() * 10000000)

        if with_mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            self._parallel = False
            self._record_on_proc = True
            self.rank = comm.rank
            self.size = comm.size
        else:
            self.rank = -1
            self.size = 1
            self._parallel = False

        if clean_up:
            with self.mongodb() as db:
                if with_mpi:
                    if comm.rank == 0:
                        delete = True
                    else:
                        delete = False
                else:
                    delete = True

                if delete:
                    db.driver_iterations.delete_many({'case_id': case_id})
                    db.solver_iterations.delete_many({'case_id': case_id})
                    db.system_iterations.delete_many({'case_id': case_id})
                    db.problem_cases.delete_many({'case_id': case_id})
                    db.global_iterations.delete_many({'case_id': case_id})
                    db.metadata.delete_many({'case_id': case_id})
                    db.driver_derivatives.delete_many({'case_id': case_id})
                    db.solver_metadata.delete_many({'case_id': case_id})
                    db.system_metadata.delete_many({'case_id': case_id})
                    db.driver_metadata.delete_many({'case_id': case_id})

        super(MongoRecorder, self).__init__(record_viewer_data)
        if with_mpi:
            self._parallel = False
            self._record_on_proc = True
            self.rank = comm.rank
            self.size = comm.size
        else:
            self.rank = -1
            self.size = 1
            self._parallel = False

        # testing
        self.iteration_coordinate_lst = []

    def get(self, key):
        with self.mongodb() as db:
            res = list(db.driver_iterations.find({'case_id': self.case_id}, {key: 1}))
        return np.asarray([x[key] for x in res])

    def __getitem__(self, key):
        return self.get(key)

    def keys(self):
        with self.mongodb() as db:
            return list(db.driver_iterations.find_one())

    def _initialize_database(self):
        """
        Initialize the database.
        """
        # TODO

        self._database_initialized = True

    def mongodb(self):
        if self.ssh_tunnel:
            print('Warning: SSH database access not available in this Topfarm version!')
            # return EC_Mongo()
        else:
            client = RemoteMongo(uri=self._uri, host=self._host, port=self._port, ip=self._ip, user=self._user, password=self._password, db_name=self.db_name, uri_type=self._uri_type, **self._client_args)
            return client

    def exists(self):
        with self.mongodb() as db:
            return db.driver_iterations.find_one({'case_id': self.case_id}) is not None

    @property
    def num_cases(self):
        return len(self.iteration_coordinate_lst)

    def _cleanup_abs2meta(self):
        """
        Convert all abs2meta variable properties to a form that can be dumped as JSON.
        """
        for name in self._abs2meta:
            for prop in self._abs2meta[name]:
                self._abs2meta[name][prop] = make_serializable(self._abs2meta[name][prop])

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
                var_settings[name][prop] = make_serializable(var_settings[name][prop])
        return var_settings

    def startup(self, recording_requester, comm=None):
        """
        Prepare for a new run and create/update the abs2prom and prom2abs variables.

        Parameters
        ----------
        recording_requester : object
            Object to which this recorder is attached.
        """

        # we only want to set up recording once for each recording_requester
        if recording_requester in self._started:
            return

        super(MongoRecorder, self).startup(recording_requester)

        if not self._database_initialized:
            self._initialize_database()

        driver = None

        # grab the system and driver
        if isinstance(recording_requester, Driver):
            system = recording_requester._problem().model
            driver = recording_requester
        # elif isinstance(recording_requester, System):
        #     system = recording_requester
        #     driver = None
        # elif isinstance(recording_requester, Problem):
        #     system = recording_requester.model
        #     driver = recording_requester.driver
        # elif isinstance(recording_requester, Solver):
        #     system = recording_requester._system()
        #     driver = None
        else:
            raise ValueError('Driver encountered a recording_requester it cannot handle'
                             ': {0}'.format(recording_requester))

        states = system._list_states_allprocs()

        if self.connection:

            if driver is None:
                pass
                # desvars = system.get_design_vars(True, get_sizes=False, use_prom_ivc=False)
                # responses = system.get_responses(True, get_sizes=False)
                # constraints = OrderedDict()
                # objectives = OrderedDict()
                # for name, data in responses.items():
                #     if data['type'] == 'con':
                #         constraints[name] = data
                #     else:
                #         objectives[name] = data
            else:
                desvars = driver._designvars.copy()
                responses = driver._responses.copy()
                constraints = driver._cons.copy()
                objectives = driver._objs.copy()

            inputs = list(system.abs_name_iter('input', local=False, discrete=True))
            outputs = list(system.abs_name_iter('output', local=False, discrete=True))

            # var_order = system._get_vars_exec_order(inputs=True, outputs=True)

            # merge current abs2prom and prom2abs with this system's version
            self._abs2prom['input'].update(system._var_allprocs_abs2prom['input'])
            self._abs2prom['output'].update(system._var_allprocs_abs2prom['output'])
            for v, abs_names in system._var_allprocs_prom2abs_list['input'].items():
                if v not in self._prom2abs['input']:
                    self._prom2abs['input'][v] = abs_names
                else:
                    self._prom2abs['input'][v] = list(set(chain(self._prom2abs['input'][v],
                                                                abs_names)))

            # for outputs, there can be only one abs name per promoted name
            for v, abs_names in system._var_allprocs_prom2abs_list['output'].items():
                self._prom2abs['output'][v] = abs_names

            # absolute pathname to metadata mappings for continuous & discrete variables
            # discrete mapping is sub-keyed on 'output' & 'input'
            real_meta_in = system._var_allprocs_abs2meta['input']
            real_meta_out = system._var_allprocs_abs2meta['output']
            disc_meta_in = system._var_allprocs_discrete['input']
            disc_meta_out = system._var_allprocs_discrete['output']

            full_var_set = [(outputs, 'output'),
                            (desvars, 'desvar'), (responses, 'response'),
                            (objectives, 'objective'), (constraints, 'constraint')]

            for var_set, var_type in full_var_set:
                for name in var_set:

                    # Design variables, constraints and objectives can be requested by input name.
                    if var_type != 'output':
                        try:
                            name = var_set[name]['ivc_source']
                        except KeyError:
                            name = var_set[name]['source']

                    if name not in self._abs2meta:
                        try:
                            self._abs2meta[name] = real_meta_out[name].copy()
                        except KeyError:
                            self._abs2meta[name] = disc_meta_out[name].copy()
                        self._abs2meta[name]['type'] = []
                        self._abs2meta[name]['explicit'] = name not in states

                    if var_type not in self._abs2meta[name]['type']:
                        self._abs2meta[name]['type'].append(var_type)

            for name in inputs:
                try:
                    self._abs2meta[name] = real_meta_in[name].copy()
                except KeyError:
                    self._abs2meta[name] = disc_meta_in[name].copy()
                self._abs2meta[name]['type'] = ['input']
                self._abs2meta[name]['explicit'] = True

            # merge current abs2meta with this system's version
            for name, meta in self._abs2meta.items():
                for io in ('input', 'output'):
                    if name in system._var_allprocs_abs2meta[io]:
                        meta.update(system._var_allprocs_abs2meta[io][name])
                        break

        self._cleanup_abs2meta()

        # store the updated abs2prom and prom2abs
        abs2prom = json.dumps(self._abs2prom)
        prom2abs = json.dumps(self._prom2abs)
        abs2meta = json.dumps(self._abs2meta)

        var_settings = {}
        var_settings.update(desvars)
        var_settings.update(objectives)
        var_settings.update(constraints)
        var_settings = self._cleanup_var_settings(var_settings)
        var_settings_json = json.dumps(var_settings)

        with self.mongodb() as db:
            data = {
                'case_id': self.case_id,
                'abs2prom': abs2prom,
                'prom2abs': prom2abs,
                'abs2meta': abs2meta,
                'var_settings_json': var_settings_json}
            db.metadata.insert_one(data)
        self._started.add(recording_requester)

    def record_iteration(self, recording_requester, data, metadata, **kwargs):
        """
        Route the record_iteration call to the proper method.

        Parameters
        ----------
        recording_requester : object
            System, Solver, Driver in need of recording.
        metadata : dict, optional
            Dictionary containing execution metadata.
        data : dict
            Dictionary containing desvars, objectives, constraints, responses, and System vars.
        **kwargs : keyword args
            Some implementations of record_iteration need additional args.
        """
        self._counter += 1

        self._iteration_coordinate = \
            recording_requester._recording_iter.get_formatted_iteration_coordinate()

        if isinstance(recording_requester, Driver):
            self.record_iteration_driver(recording_requester, data, metadata)
        # elif isinstance(recording_requester, System):
        #     self.record_iteration_system(recording_requester, data, metadata)
        # elif isinstance(recording_requester, Solver):
        #     self.record_iteration_solver(recording_requester, data, metadata)
        # elif isinstance(recording_requester, Problem):
        #     self.record_iteration_problem(recording_requester, data, metadata)
        else:
            raise ValueError("Recorders must be attached to Drivers, Systems, or Solvers.")

    def record_iteration_driver(self, driver, data, metadata):
        """
        Record data and metadata from a Driver.

        Parameters
        ----------
        driver : Driver
            Driver in need of recording.
        data : dict
            Dictionary containing desvars, objectives, constraints, responses, and System vars.
        metadata : dict
            Dictionary containing execution metadata.
        """
        if not self._database_initialized:
            raise RuntimeError(f"{driver.msginfo} attempted to record iteration to "
                               f"'{self._filepath}', but database is not initialized;"
                               " `run_model()`, `run_driver()`, or `final_setup()` "
                               "must be called after adding a recorder.")

        if self.connection:
            outputs = data['output']
            outputs = {self._abs2prom['output'][k]: v for k, v in outputs.items()}
            inputs = data['input']
            inputs = {self._abs2prom['input'][k]: v for k, v in inputs.items()}
            residuals = data['residual']

            # convert to list so this can be dumped as JSON
            for in_out_resid in (inputs, outputs, residuals):
                if in_out_resid is None:
                    continue
                for var in in_out_resid:
                    in_out_resid[var] = make_serializable(in_out_resid[var])

            # outputs_text = json.dumps(outputs)
            # inputs_text = json.dumps(inputs)
            # residuals_text = json.dumps(residuals)

            if isinstance(driver, RandomSearchDriver):
                max_step = driver.randomize_func.max_step
            else:
                max_step = None

            with self.mongodb() as db:
                self.lastrowid += 1
                data = {
                    'case_id': self.case_id,
                    'run_id': self.run_id,
                    'counter': self._counter,
                    'timestamp': metadata['timestamp'],
                    'success': metadata['success'],
                    'msg': metadata['msg'],
                    'rank': self.rank,
                    'size': self.size,
                    'max_step': max_step
                }
                data.update(make_serializable(inputs))
                data.update(make_serializable(outputs))
                db.driver_iterations.insert_one(data)

                data = {
                    'case_id': self.case_id,
                    'run_id': self.run_id,
                    'counter': self._counter,
                    'record_type': 'driver',
                    'rowid': self.lastrowid,
                    'source': driver._get_name()
                }
                db.global_iterations.insert_one(data)

    def record_iteration_problem(self, data, metadata):
        """
        Record data and metadata from a Problem.

        Parameters
        ----------
        recording_requester : object
            Problem in need of recording.
        data : dict
            Dictionary containing desvars, objectives, and constraints.
        metadata : dict
            Dictionary containing execution metadata.
        """
        pass
        # outputs = data['output']

        # # convert to list so this can be dumped as JSON
        # if outputs is not None:
        #     for var in outputs:
        #         outputs[var] = make_serializable(outputs[var])

        # with self.mongodb() as db:
        #     data = {
        #         'case_id': self.case_id,
        #         'counter': self._counter,
        #         'timestamp': metadata['timestamp'],
        #         'success': metadata['success'],
        #         'msg': metadata['msg'],
        #     }
        #     data.update(make_serializable(outputs))
        #     db.problem_cases.inser_one(data)

    def record_iteration_system(self, recording_requester, data, metadata):
        """
        Record data and metadata from a System.

        Parameters
        ----------
        recording_requester : System
            System in need of recording.
        data : dict
            Dictionary containing inputs, outputs, and residuals.
        metadata : dict
            Dictionary containing execution metadata.
        """
        pass
        # inputs = data['i']
        # outputs = data['o']
        # residuals = data['r']

        # # convert to list so this can be dumped as JSON
        # for i_o_r in (inputs, outputs, residuals):
        #     if i_o_r is None:
        #         continue
        #     for var in i_o_r:
        #         i_o_r[var] = make_serializable(i_o_r[var])

        # with self.mongodb() as db:
        #     data = {
        #         'case_id': self.case_id,
        #         'iteration_coordinate': self._iteration_coordinate,
        #         'counter': self._counter,
        #         'timestamp': metadata['timestamp'],
        #         'success': metadata['success'],
        #         'msg': metadata['msg'],
        #         'residuals': make_serializable(residuals)
        #     }
        #     data.update(make_serializable(inputs))
        #     data.update(make_serializable(outputs))
        #     db.system_iterations.inser_one(data)

        #     # get the pathname of the source system
        #     source_system = recording_requester.pathname
        #     if source_system == '':
        #         source_system = 'root'

        #     data = {
        #         'case_id': self.case_id,
        #         'counter': self._counter,
        #         'record_type': 'system',
        #         'rowid': self.lastrowid,
        #         'source': source_system
        #     }
        #     db.global_iterations.insert_one(data)

    def record_iteration_solver(self, recording_requester, data, metadata):
        """
        Record data and metadata from a Solver.

        Parameters
        ----------
        recording_requester : Solver
            Solver in need of recording.
        data : dict
            Dictionary containing outputs, residuals, and errors.
        metadata : dict
            Dictionary containing execution metadata.
        """
        pass
        # abs = data['abs']
        # rel = data['rel']
        # inputs = data['i']
        # outputs = data['o']
        # residuals = data['r']

        # with self.mongodb() as db:
        #     data = {
        #         'case_id': self.case_id,
        #         'iteration_coordinate': self._iteration_coordinate,
        #         'counter': self._counter,
        #         'timestamp': metadata['timestamp'],
        #         'success': metadata['success'],
        #         'msg': metadata['msg'],
        #         'abs_err': abs,
        #         'rel_err': rel,
        #         'residuals': make_serializable(residuals)
        #     }
        #     data.update(make_serializable(inputs))
        #     data.update(make_serializable(outputs))
        #     db.solver_iterations.inser_one(data)

        #     # get the pathname of the source system
        #     source_system = recording_requester._system.pathname
        #     if source_system == '':
        #         source_system = 'root'

        #     solver_type = recording_requester.SOLVER[0:2]
        #     if solver_type == 'NL':
        #         source_solver = source_system + '.nonlinear_solver'
        #     elif solver_type == 'LS':
        #         source_solver = source_system + '.nonlinear_solver.linesearch'
        #     else:
        #         raise RuntimeError("Solver type '%s' not recognized during recording. "
        #                            "Expecting NL or LS" % recording_requester.SOLVER)

        #     data = {
        #         'case_id': self.case_id,
        #         'counter': self._counter,
        #         'record_type': 'solver',
        #         'rowid': self.lastrowid,
        #         'source': source_solver
        #     }
        #     db.global_iterations.insert_one(data)

    def record_viewer_data(self, model_viewer_data, key='Driver'):
        """
        Record model viewer data.

        Parameters
        ----------
        model_viewer_data : dict
            Data required to visualize the model.
        key : str, optional
            The unique ID to use for this data in the table.
        """
        pass
        # json_data = json.dumps(model_viewer_data, default=make_serializable)
        # with self.mongodb() as db:
        #     data = {
        #         'case_id': self.case_id,
        #         'id': key,
        #         'model_viewer_data': json_data
        #     }
        #     db.driver_metadata.insert_one(data)

    def record_metadata_system(self, system, run_number=None):
        """
        Record system metadata.

        Parameters
        ----------
        system : System
            The System for which to record metadata.
        run_number : int or None
            Number indicating which run the metadata is associated with.
            None for the first run, 1 for the second, etc.
        """
        pass
        # if self._record_metadata and self.metadata_connection:

        #     scaling_vecs, user_options = self._get_metadata_system(system)

        #     if scaling_vecs is None:
        #         return

        #     path = system.pathname
        #     if not path:
        #         path = 'root'

        #     if run_number is None:
        #         name = path
        #     else:
        #         name = META_KEY_SEP.join([path, str(run_number)])

        #     with self.mongodb() as db:
        #         data = {
        #             'case_id': self.case_id,
        #             'name': name,
        #             'scaling_factors': scaling_vecs,
        #             'component_metadata': user_options
        #         }
        #         db.system_metadata.inser_one(data)

    def record_metadata_solver(self, solver, run_number=None):
        """
        Record solver metadata.

        Parameters
        ----------
        solver : Solver
            The Solver for which to record metadata.
        run_number : int or None
            Number indicating which run the metadata is associated with.
            None for the first run, 1 for the second, etc.
        """
        pass
        # if self._record_metadata and self.metadata_connection:
        #     path = solver._system().pathname
        #     solver_class = type(solver).__name__

        #     if not path:
        #         path = 'root'

        #     id = "{}.{}".format(path, solver_class)

        #     if run_number is not None:
        #         id = META_KEY_SEP.join([id, str(run_number)])

        #     with self.mongodb() as db:
        #         data = {
        #             'case_id': self.case_id,
        #             'id': id,
        #             'solver_class': solver_class
        #         }
        #         db.solver_metadata.inser_one(data)

    def shutdown(self):
        """
        Shut down the recorder.
        """

    def animate_turbineXY(self, duration=10, tail=5, cost='aep', anim_options=None, filename=None):
        """
        Display turbine XY positions between iterations as an animation.

        Parameters
        ----------
        duration : float

        tail : int
            Amount of iterations to show on on the trailing tail.
        cost : str
            name of cost function to be displayed as in MongoDB.
            Defaults to 'aep'.
        filename : str
            Name to save .mp4 animation as.
            Set to "None" to display animation with current Matplotlib backend.
            Defaults to "None".
        Returns
        -------
        ani : Object
            If "filename" exists the method will return a "filename.mp4" file
            Otherwise a Matplotlib animation object is returned.

        """
        x = self['x']
        y = self['y']
        cost = self[cost]
        n_wt = x.shape[1]
        boundary = self['xy_boundary'][-1]
        boundary = np.r_[boundary, boundary[:1]]

        # Setup figure
        fig, ax = plt.subplots()
        # TODO Implement an init() function with axis limits and setting of boundaries
        ax.set_xlim([np.min([np.min(x), np.min(boundary[:, 0])]), np.max([np.max(x), np.max(boundary[:, 0])])])
        ax.set_ylim([np.min([np.min(y), np.min(boundary[:, 1])]), np.max([np.max(y), np.max(boundary[:, 1])])])
        ax.plot(boundary[:, 0], boundary[:, 1])
        # Initialize "ln" list to be updated during animation
        colors = [c['color'] for c in iter(mpl.rcParams['axes.prop_cycle'])] * 100
        ln = [plt.plot([], [], '.-', ms=10, color=c, animated=True)[0] for c in colors[:n_wt]]
        ln += [plt.plot([], [], '--', color=c, animated=True)[0] for c in colors[:n_wt]]
        ln += [plt.plot([], [], 'xk', ms=4, animated=True)[0] for _ in colors[:n_wt]]
        title = ax.text(0.5, .95, "", bbox={'facecolor': 'w', 'alpha': .8, 'pad': 5},
                        transform=ax.transAxes, ha="center")
        ln += [title]

        def update(frame):
            """
            Update data in "ln" for every frame in FuncAnimation().
            """
            title.set_text("%f (%.2f%%)" % (cost[frame],
                                            (cost[frame] - cost[0]) / cost[0] * 100))
            for i in range(n_wt):
                ln[i].set_data(x[max(0, frame - tail):frame + 1, i], y[max(0, frame - tail):frame + 1, i])
                ln[i + n_wt].set_data(np.r_[x[0, i], x[frame, i]], np.r_[y[0, i], y[frame, i]])
                ln[i + 2 * n_wt].set_data(x[frame, i], y[frame, i])
            return ln

        # Run animation
        if anim_options is None:
            ani = animation.FuncAnimation(fig, update, blit=True, frames=x.shape[0], interval=100, repeat=False)
        else:
            ani = animation.FuncAnimation(fig, update, frames=x.shape[0], **anim_options)

        # Exit behaviour decided by presence of a "filename"
        if filename:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=len(x) / duration, metadata=dict(artist='TopFarm2', title='Turbine XY - Animation'), bitrate=1800)
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            ani.save(filename + ".mp4", writer=writer)
        else:
            return ani
