from abc import ABC, abstractmethod
from openmdao.core.explicitcomponent import ExplicitComponent


class Constraint(ABC):

    @abstractmethod
    def setup_as_constraint(self):
        pass

    @abstractmethod
    def setup_as_penalty(self):
        pass

    def _setup_as_penalty(self, problem, name, comp, setup, penalty_func):
        # subsystem_order = [ss.name for ss in problem.model._static_subsystems_allprocs]
        problem.model.add_subsystem(name, comp, promotes=['*'])
        # if 'cost_comp' in subsystem_order:  # penalty comp must be setup before cost
        # subsystem_order.insert(subsystem_order.index('cost_comp'), name)
        # problem.model.set_order(subsystem_order)

        self._cost_comp = problem.cost_comp
        self._org_setup = self._cost_comp.setup
        self._org_compute = self._cost_comp.compute

        def new_setup():
            self._org_setup()
            setup()
        self._cost_comp.setup = new_setup

        def new_compute(inputs, outputs):
            p = penalty_func(inputs)
            if p == 0:
                self._org_compute(inputs, outputs)
            else:
                outputs['cost'] = 1e10 + p
        self._cost_comp.compute = new_compute


class ConstraintComponent(ExplicitComponent, ABC):
    def __init__(self, **kwargs):
        ExplicitComponent.__init__(self, **kwargs)

    @abstractmethod
    def satisfy(self, state):
        pass

    def plot(self, ax):
        pass
