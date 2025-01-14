from abc import ABC, abstractmethod
from openmdao.core.explicitcomponent import ExplicitComponent


class Constraint(ABC):

    @abstractmethod
    def setup_as_constraint(self):  # pragma: no cover
        pass

    @abstractmethod
    def setup_as_penalty(self):  # pragma: no cover
        pass

    @property
    def constraintComponent(self):
        return self.comp


class ConstraintComponent(ExplicitComponent, ABC):
    def __init__(self, **kwargs):
        ExplicitComponent.__init__(self, **kwargs)

    @abstractmethod
    def satisfy(self, state):  # pragma: no cover
        pass

    def plot(self, ax):
        pass
