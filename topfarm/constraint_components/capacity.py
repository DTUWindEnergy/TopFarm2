import numpy as np
from topfarm.constraint_components import Constraint, ConstraintComponent
import topfarm


class CapacityConstraint(Constraint):
    def __init__(self, max_capacity=500, rated_power_array=[3000]):
        """Initialize CapacityConstraint

        Parameters
        ----------
        max_capacity : int or float
            Maximum wind farm installed capacity [MW]
        rated_power_array : list
            Rated power list corresponding to turbine type order [kW]
        """
        self.max_capacity = max_capacity
        self.rated_power_array = np.array(rated_power_array) / 1e3
        self.const_id = 'capacity_comp_{}'.format(int(max_capacity))

    @property
    def constraintComponent(self):
        return self.capacity_comp

    def _setup(self, problem):
        self.n_wt = problem.n_wt
        self.capacity_comp = CapacityComp(self.n_wt, self.max_capacity, self.rated_power_array, self.const_id)
        problem.model.constraint_group.add_subsystem(self.const_id, self.capacity_comp, promotes=[
            '*'])

    def setup_as_constraint(self, problem):
        self._setup(problem)
        problem.model.add_constraint('totalcapacity', upper=self.max_capacity)

    def setup_as_penalty(self, problem):
        self._setup(problem)


class CapacityComp(ConstraintComponent):
    """
    Calculates total installed capacity of the wind farm.
    """

    def __init__(self, n_wt, max_capacity, rated_power_array, const_id=None):
        super().__init__()
        self.n_wt = n_wt
        self.max_capacity = max_capacity
        self.rated_power_array = np.array(rated_power_array)
        self.const_id = const_id

    def setup(self):
        self.add_input(topfarm.type_key, np.zeros(self.n_wt, dtype=int))
        # self.add_output('constraint_violation_' + self.const_id, val=0.0)
        self.add_output('totalcapacity', val=0.0,
                        desc='wind farm installed capacity')
        # Partial declaration is not needed for type or penalty in this case.
        # Because it is only used for integer optimization.

    def compute(self, inputs, outputs):
        outputs['totalcapacity'] = np.sum(self.rated_power_array[inputs[topfarm.type_key].astype(int)])
        # outputs['constraint_violation_' + self.const_id] = np.maximum(0, outputs['totalcapacity'][0] - self.max_capacity)

    def satisfy(self):
        pass
