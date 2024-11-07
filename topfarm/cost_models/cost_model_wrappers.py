from openmdao.core.explicitcomponent import ExplicitComponent
import numpy as np
import time
from _collections import defaultdict
# from topfarm.constraint_components.post_constraint import PostConstraint
import warnings


class CostModelComponent(ExplicitComponent):
    """Wrapper for pure-Python cost functions"""

    def __init__(self, input_keys, n_wt, cost_function, cost_gradient_function=None,
                 output_keys=["Cost"], output_unit="", additional_input=[], additional_output=[], max_eval=None,
                 objective=True, maximize=False, output_vals=[0.0], input_units=[], step={}, use_constraint_violation=True, **kwargs):
        """Initialize wrapper for pure-Python cost function

        Parameters
        ----------
        input_keys : list of str
            Inputs to the cost function
        n_wt : int
            Number of wind turbines
        cost_function : function handle
            Function to evaluate cost
        cost_gradient_function : function handle, optional
            Function to evaluate gradient of the cost function
        output_key : list
            Name of output key
        output_unit : str
            Units of output of cost function
        additional_input : list of str
            Other (non-design-variable) inputs required by the cost function\n
            Gradients will not be computed for these inputs
        additional_output : list of str or list of tuples
            Other outputs to request\n
            if list of str: ['add_out_name',...]\n
            if list of tuples [('add_out_name', val),...], where val is a template value of the output
            The cost function must return: cost, {'add_out1_name': add_out1, ...}
        max_eval : int
            Maximum number of function evaluations
        objective : boolean
            Must be True for standalone CostModelComponents and the final component in a TopFarmGroup
        maximize : boolean
            If True: objective is maximised during optimization\n
            If False: Objective is minimized during optimization
        output_vals : float or array_like
            Format of output
        input_units : list of str
            Units of the respective input_keys
        step : dict of {str : float}
            Finite difference step size for each input key, e.g. {input_key : step_size, input_key2 : step_size2}
        """
        if 'income_model' in kwargs:
            warnings.warn("""income_model is deprecated; use keyword maximize instead""",
                          DeprecationWarning, stacklevel=2)
            maximize = kwargs['income_model']
            kwargs.pop('income_model')
        if 'output_key' in kwargs:
            warnings.warn("""output_key is deprecated; use keyword output_keys instead""",
                          DeprecationWarning, stacklevel=2)
            output_keys = [kwargs['output_key']]
            kwargs.pop('output_key')
        else:
            self.output_key = output_keys[0]
        if 'output_val' in kwargs:
            warnings.warn("""output_val is deprecated; use keyword output_vals instead""",
                          DeprecationWarning, stacklevel=2)
            output_vals = [kwargs['output_val']]
            kwargs.pop('output_val')
        super().__init__(**kwargs)
        assert isinstance(n_wt, int), n_wt
        self.input_keys = list(input_keys)
        self.input_keys_only = list([(i, i[0])[isinstance(i, tuple)] for i in self.input_keys])
        self.additional_input = additional_input
        self.all_input_keys = self.input_keys_only + list([(i, i[0])[isinstance(i, tuple)] for i in self.additional_input])
        self.cost_function = cost_function
        self.cost_gradient_function = cost_gradient_function
        self.n_wt = n_wt
        if not isinstance(output_keys, list):
            output_keys = [output_keys]
        if not isinstance(output_vals, list):
            output_vals = [output_vals]
        self.output_keys = output_keys
        self.out_keys_only = list([(o, o[0])[isinstance(o, tuple)] for o in self.output_keys])
        self.output_unit = output_unit
        self.output_key = output_keys[0]
        if isinstance(self.output_key, tuple):
            self.output_key = self.output_key[0]
        self.additional_output = [((x, np.zeros(self.n_wt)), x)[isinstance(x, tuple)] for x in additional_output]
        self.max_eval = max_eval or 1e100
        self.objective = objective
        if maximize:
            self.cost_factor = -1.0
        else:
            self.cost_factor = 1.0
        output_vals = output_vals[:len(output_keys)] + [output_vals[0]] * (len(output_keys) - len(output_vals))  # extend output_vals if it is not same length as output_keys
        self.output_vals = output_vals

        n_input = len(self.input_keys) + len(additional_input)
        self.input_units = (input_units + [None] * n_input)[:n_input]

        self.n_func_eval = 0
        self.func_time_sum = 0
        self.n_grad_eval = 0
        self.grad_time_sum = 0
        self.step = step
        self.use_constraint_violation = use_constraint_violation

    def setup(self):
        for i, u in zip(self.input_keys + self.additional_input, self.input_units):
            if isinstance(i, tuple) and len(i) == 2:
                self.add_input(i[0], val=i[1], units=u)
            else:
                self.add_input(i, val=np.zeros(self.n_wt), units=u)
        if self.use_constraint_violation:
            self.add_input('constraint_violation', val=0.0)
        if self.objective:
            self.add_output('cost', val=0.0)
            self.add_output('cost_comp_eval', val=0.0)
        for o, v in zip(self.output_keys, self.output_vals):
            if isinstance(o, tuple) and len(o) == 2:
                self.add_output(o[0], val=o[1])
            else:
                self.add_output(o, val=v)
        for key, val in self.additional_output:
            self.add_output(key, val=val)

        if self.cost_gradient_function:
            if self.objective:
                self.declare_partials('cost', self.input_keys_only, method='exact')
            for o in self.out_keys_only:
                self.declare_partials(o, self.input_keys_only, method='exact')
        else:
            if self.step == {}:
                if self.objective:
                    self.declare_partials('cost', self.input_keys_only, method='fd')
                for o in self.out_keys_only:
                    self.declare_partials(o, self.input_keys_only, method='fd')
            else:
                for i in self.input_keys_only:
                    if self.objective:
                        self.declare_partials('cost', i, step=self.step[i], method='fd')
                    for o in self.out_keys_only:
                        self.declare_partials(o, i, step=self.step[i], method='fd')

    @property
    def counter(self):
        counter = float(self.n_func_eval)
        if self.grad_time_sum > 0 and self.func_time_sum > 0 and self.n_grad_eval > 0 and self.n_func_eval > 0:
            ratio = ((self.grad_time_sum / self.n_grad_eval) /
                     (self.func_time_sum / self.n_func_eval))
            counter += self.n_grad_eval * max(ratio, 1)
        else:
            counter += self.n_grad_eval
        return int(counter)

    def compute(self, inputs, outputs):
        """Compute cost model"""
        if self.use_constraint_violation:
            if (inputs['constraint_violation'] > 1e5) and (self.n_func_eval > 0):
                return
        if self.counter >= self.max_eval:
            return
        t = time.time()
        if self.additional_output:
            c, additional_output = self.cost_function(**{x: inputs[x] for x in self.all_input_keys})
            for k, v in additional_output.items():
                outputs[k] = v
        else:
            c = self.cost_function(**{x: inputs[x] for x in self.all_input_keys})
        if not isinstance(c, list):
            c = [c]
        if self.objective:
            outputs['cost'] = c[0] * self.cost_factor
            outputs['cost_comp_eval'] = self.counter
        for o, _c in zip(self.out_keys_only, c):
            outputs[o] = _c
        self.func_time_sum += time.time() - t
        self.n_func_eval += 1

    def compute_partials(self, inputs, J):
        if self.counter >= self.max_eval:
            return

        if hasattr(self, 'skip_linearize'):
            if self.skip_linearize:
                return

        t = time.time()
        if self.cost_gradient_function:
            for k, dCostdk in zip(self.input_keys_only,
                                  self.cost_gradient_function(**{x: inputs[x] for x in self.all_input_keys})):
                if dCostdk is not None:
                    if not isinstance(dCostdk, list):
                        dCostdk = [dCostdk]
                    if self.objective:
                        J['cost', k] = dCostdk[0] * self.cost_factor
                    for o, _d in zip(self.out_keys_only, dCostdk):
                        J[o, k] = _d
        self.grad_time_sum += time.time() - t
        self.n_grad_eval += 1


class AEPCostModelComponent(CostModelComponent):
    """Wrapper for pure-Python cost functions"""

    def __init__(self, input_keys, n_wt, cost_function, cost_gradient_function=None,
                 output_unit="", additional_input=[], additional_output=[], max_eval=None,
                 output_key="AEP", **kwargs):
        """Initialize Sub class of the CostModelComponent for AEP maximization

        Parameters
        ----------
        input_keys : list of str
            Inputs to the cost function
        n_wt : int
            Number of wind turbines
        cost_function : function handle
            Function to evaluate cost
        cost_gradient_function : function handle, optional
            Function to evaluate gradient of the cost function
        output_key : list
            Name of output key, default is AEP
        output_unit : str
            Units of output of cost function
        additional_input : list of str
            Other (non-design-variable) inputs required by the cost function\n
            Gradients will not be computed for these inputs
        additional_output : list of str or list of tuples
            Other outputs to request\n
            if list of str: ['add_out_name',...]\n
            if list of tuples [('add_out_name', val),...], where val is a template value of the output
            The cost function must return: cost, {'add_out1_name': add_out1, ...}
        max_eval : int
            Maximum number of function evaluations
        """
        CostModelComponent.__init__(self, input_keys, n_wt, cost_function,
                                    cost_gradient_function=cost_gradient_function,
                                    output_key=output_key, output_unit=output_unit,
                                    additional_input=additional_input,
                                    additional_output=additional_output,
                                    max_eval=max_eval, maximize=True, **kwargs)


class AEPMaxLoadCostModelComponent(CostModelComponent):
    """Wrapper for pure-Python cost functions"""

    def __init__(self, input_keys, n_wt, aep_load_function, max_loads,
                 aep_load_gradient=None, output_keys=["AEP", 'loads'], step={},
                 maximize=True, **kwargs):
        """Initialize Sub class of the CostModelComponent for AEP maximization and load constraints

        Parameters
        ----------
        input_keys : list of str
            Inputs to the cost function
        n_wt : int
            Number of wind turbines
        aep_load_function : function handle
            Function to evaluate cost
        aep_load_gradient : function handle, optional
            Function to evaluate gradient of the cost function
        output_keys : list
            Name of output key, default is AEP and loads
        step : dict of {str : float}
            Finite difference step size for each input key, e.g. {input_key : step_size, input_key2 : step_size2}
        maximize : boolean
            If True: objective is maximised during optimization\n
            If False: Objective is minimized during optimization
        """

        self.max_loads = max_loads

        def cost_function(**kwargs):
            aep, load = aep_load_function(**kwargs)
            return [aep, load]

        if aep_load_gradient:
            def gradient_function(**kwargs):
                d_aep, d_load = aep_load_gradient(**kwargs)
                return [[d_aep, d_load]]
        else:
            gradient_function = None

        # additional_output = kwargs.get('additional_output', []) + [('loads', max_loads)]
        CostModelComponent.__init__(self, input_keys, n_wt, cost_function=cost_function,
                                    cost_gradient_function=gradient_function,
                                    output_keys=output_keys, step=step, maximize=maximize,
                                    **kwargs)
        self.post_constraint = ('loads', {'upper': max_loads})
        # PostConstraint.__init__(self, 'loads', upper=max_loads)

    # def setup(self):
    #     AEPCostModelComponent.setup(self)
    #     input_keys = list([(i, i[0])[isinstance(i, tuple)] for i in self.input_keys])
        # self.declare_partials('loads', input_keys, method='fd')
