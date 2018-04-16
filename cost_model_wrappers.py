from openmdao.core.explicitcomponent import ExplicitComponent
import numpy as np



class CostModelComponent(ExplicitComponent):
    def __init__(self, n_wt, cost_function, cost_gradient_function):
        super().__init__()
        self.cost_function = cost_function
        self.cost_gradient_function = cost_gradient_function
        self.n_wt = n_wt

    def setup(self):
        self.add_input('turbineX', val=np.zeros(self.n_wt), units='m')
        self.add_input('turbineY', val=np.zeros(self.n_wt), units='m')

        self.add_output('cost', val=0.0)

        # Finite difference all partials.
        self.declare_partials('cost', '*')

    def compute(self, inputs, outputs):
        x = inputs['turbineX']
        y = inputs['turbineY']
        outputs['cost'] = self.cost_function(x, y)

    def compute_partials(self, inputs, J):
        x = inputs['turbineX']
        y = inputs['turbineY']
        dCostdx, dCostdy = self.cost_gradient_function(x, y)
        J['cost', 'turbineX'] = dCostdx
        J['cost', 'turbineY'] = dCostdy


class AEPCostModelComponent(CostModelComponent):
    def compute(self, inputs, outputs):
        CostModelComponent.compute(self, inputs, outputs)
        outputs['cost'] *= -1

    def compute_partials(self, inputs, J):
        CostModelComponent.compute_partials(self, inputs, J)
        J['cost', 'turbineX'] *= -1
        J['cost', 'turbineY'] *= -1
