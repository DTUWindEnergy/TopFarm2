from openmdao.core.explicitcomponent import ExplicitComponent
import numpy as np


class CostModelComponent(ExplicitComponent):
    def __init__(self, input_keys, n_wt, cost_function, cost_gradient_function=None,
                 output_key="Cost", output_unit="", additional_output=[]):
        super().__init__()
        assert isinstance(n_wt, int), n_wt
        self.input_keys = list(input_keys)
        self.cost_function = cost_function
        self.cost_gradient_function = cost_gradient_function
        self.n_wt = n_wt
        self.output_key = output_key
        self.output_unit = output_unit
        self.additional_output = additional_output

    def setup(self):
        for i in self.input_keys:
            if isinstance(i, tuple) and len(i) == 2:
                self.add_input(i[0], val=i[1])
            else:
                self.add_input(i, val=np.zeros(self.n_wt))
        self.add_output('cost', val=0.0)
        self.add_output(self.output_key, val=0.0)
        for key, val in self.additional_output:
            self.add_output(key, val=val)

        input_keys = list([(i, i[0])[isinstance(i, tuple)] for i in self.input_keys])
        if self.cost_gradient_function:
            self.declare_partials('cost', input_keys)
        else:
            # Finite difference all partials.
            self.declare_partials('cost', input_keys, method='fd')
        self.declare_partials(self.output_key, input_keys)

    def compute(self, inputs, outputs):
        if self.additional_output:
            c, additional_output = self.cost_function(**inputs)
            for k, v in additional_output.items():
                outputs[k] = v
        else:
            c = self.cost_function(**inputs)
        outputs['cost'] = c
        outputs[self.output_key] = c

    def compute_partials(self, inputs, J):
        if self.cost_gradient_function:
            for k, dCostdk in zip(self.input_keys,
                                  self.cost_gradient_function(**inputs)):
                if dCostdk is not None:
                    J['cost', k] = dCostdk
                    J[self.output_key, k] = dCostdk


class IncomeModelComponent(CostModelComponent):

    def compute(self, inputs, outputs):
        CostModelComponent.compute(self, inputs, outputs)
        outputs['cost'] *= -1

    def compute_partials(self, inputs, J):
        if self.cost_gradient_function:
            CostModelComponent.compute_partials(self, inputs, J)
            for k in dict(inputs).keys():
                J['cost', k] *= -1


class AEPCostModelComponent(IncomeModelComponent):
    def __init__(self, input_keys, n_wt, cost_function, cost_gradient_function=None, output_unit="", additional_output=[]):
        IncomeModelComponent.__init__(self, input_keys, n_wt, cost_function,
                                      cost_gradient_function=cost_gradient_function,
                                      output_key="AEP", output_unit=output_unit,
                                      additional_output=additional_output)
