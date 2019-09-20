from openmdao.core.explicitcomponent import ExplicitComponent
import numpy as np
import time
from _collections import defaultdict


class CostModelComponent(ExplicitComponent):
    """Wrapper for pure-Python cost functions"""

    def __init__(self, input_keys, n_wt, cost_function, cost_gradient_function=None,
                 output_key="Cost", output_unit="", additional_input=[], additional_output=[], max_eval=None,
                 objective=True, income_model=False, output_val=0.0, input_units=[], **kwargs):
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
        output_key : str
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
        income_model : boolean
            If True: objective is maximised during optimization\n
            If False: Objective is minimized during optimization
        output_val : float or array_like
            Format of output
        input_units : list of str
            Units of the respective input_keys
        """
        super().__init__(**kwargs)
        assert isinstance(n_wt, int), n_wt
        self.input_keys = list(input_keys)
        self.cost_function = cost_function
        self.cost_gradient_function = cost_gradient_function
        self.n_wt = n_wt
        self.output_key = output_key
        self.output_unit = output_unit
        self.additional_input = additional_input
        self.additional_output = [((x, np.zeros(self.n_wt)), x)[isinstance(x, tuple)] for x in additional_output]
        self.max_eval = max_eval or 1e100
        self.objective = objective
        if income_model:
            self.cost_factor = -1.0
        else:
            self.cost_factor = 1.0
        self.output_val = output_val

        n_input = len(self.input_keys) + len(additional_input)
        self.input_units = (input_units + [None] * n_input)[:n_input]

        self.n_func_eval = 0
        self.func_time_sum = 0
        self.n_grad_eval = 0
        self.grad_time_sum = 0
        self.step = {}

    def setup(self):
        for i, u in zip(self.input_keys + self.additional_input, self.input_units):
            if isinstance(i, tuple) and len(i) == 2:
                self.add_input(i[0], val=i[1], units=u)
            else:
                self.add_input(i, val=np.zeros(self.n_wt), units=u)
        self.add_input('penalty', val=0.0)
        if self.objective:
            self.add_output('cost', val=0.0)
            self.add_output('cost_comp_eval', val=0.0)
        self.add_output(self.output_key, val=self.output_val)
        for key, val in self.additional_output:
            self.add_output(key, val=val)

        input_keys = list([(i, i[0])[isinstance(i, tuple)] for i in self.input_keys])
        self.inp_keys = input_keys + list([(i, i[0])[isinstance(i, tuple)] for i in self.additional_input])
        self.input_keys = input_keys
        if self.objective:
            if self.cost_gradient_function:
                self.declare_partials('cost', input_keys)
            else:
                # Finite difference all partials.
                if self.step == {}:
                    self.declare_partials('cost', input_keys, method='fd')
                else:
                    for i in input_keys:
                        self.declare_partials('cost', i, step=self.step.get('cost', {}).get(i, None), method='fd')
            self.declare_partials(self.output_key, input_keys)
        else:
            if self.cost_gradient_function:
                self.declare_partials(self.output_key, input_keys)
            else:
                # Finite difference all partials.
                self.declare_partials(self.output_key, input_keys, method='fd')

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
        if inputs['penalty'] > 0:
            return
        if self.counter >= self.max_eval:
            return
        t = time.time()
        if self.additional_output:
            c, additional_output = self.cost_function(**{x: inputs[x] for x in self.inp_keys})
            for k, v in additional_output.items():
                outputs[k] = v
        else:
            c = self.cost_function(**{x: inputs[x] for x in self.inp_keys})
        if self.objective:
            outputs['cost'] = c * self.cost_factor
            outputs['cost_comp_eval'] = self.counter
        outputs[self.output_key] = c
        self.func_time_sum += time.time() - t
        self.n_func_eval += 1

    def compute_partials(self, inputs, J):
        if self.counter >= self.max_eval:
            return

        t = time.time()
        if self.cost_gradient_function:
            for k, dCostdk in zip(self.input_keys,
                                  self.cost_gradient_function(**{x: inputs[x] for x in self.inp_keys})):
                if dCostdk is not None:
                    if self.objective:
                        J['cost', k] = dCostdk * self.cost_factor
                    J[self.output_key, k] = dCostdk
        self.grad_time_sum += time.time() - t
        self.n_grad_eval += 1


class AEPCostModelComponent(CostModelComponent):
    def __init__(self, input_keys, n_wt, cost_function, cost_gradient_function=None,
                 output_unit="", additional_input=[], additional_output=[], max_eval=None):
        CostModelComponent.__init__(self, input_keys, n_wt, cost_function,
                                    cost_gradient_function=cost_gradient_function,
                                    output_key="AEP", output_unit=output_unit,
                                    additional_input=additional_input, additional_output=additional_output,
                                    max_eval=max_eval, income_model=True)
