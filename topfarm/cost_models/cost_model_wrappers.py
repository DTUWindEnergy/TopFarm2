from openmdao.core.explicitcomponent import ExplicitComponent
import numpy as np


class CostModelComponent(ExplicitComponent):
    def __init__(self, input_keys, n_wt, cost_function, cost_gradient_function=None, input_units={'turbine' + xyz: 'm' for xyz in 'XYZ'}):
        super().__init__()
        assert isinstance(n_wt, int), n_wt
        self.input_keys = input_keys
        self.cost_function = cost_function
        self.cost_gradient_function = cost_gradient_function
        self.input_units = input_units
        self.n_wt = n_wt

    def setup(self):
        for i in self.input_keys:
            self.add_input(i, val=np.zeros(self.n_wt), units=self.input_units.get(i, None))
        self.add_output('cost', val=0.0)

        if self.cost_gradient_function:
            self.declare_partials('cost', self.input_keys)
        else:
            # Finite difference all partials.
            self.declare_partials('cost', self.input_keys, method='fd')

    def compute(self, inputs, outputs):
        outputs['cost'] = self.cost_function(**inputs)

    def compute_partials(self, inputs, J):
        if self.cost_gradient_function:
            for k, dCostdk in zip(self.input_keys,
                                  self.cost_gradient_function(**inputs)):
                if dCostdk is not None:
                    J['cost', k] = dCostdk


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
    pass
