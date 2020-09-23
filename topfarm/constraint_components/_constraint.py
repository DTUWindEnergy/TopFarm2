from abc import ABC, abstractmethod
from openmdao.core.explicitcomponent import ExplicitComponent


class Constraint(ABC):

    @abstractmethod
    def setup_as_constraint(self):
        pass

    @abstractmethod
    def setup_as_penalty(self):
        pass

    @property
    def constraintComponent(self):
        return self.comp


class ConstraintComponent(ExplicitComponent, ABC):
    def __init__(self, **kwargs):
        ExplicitComponent.__init__(self, **kwargs)

    @abstractmethod
    def satisfy(self, state):
        pass

    def plot(self, ax):
        pass
