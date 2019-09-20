from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
import numpy as np


class PenaltyComponent(ExplicitComponent):
    def __init__(self, constraint_comps, as_penalty):
        super().__init__()
        self.constraint_comps = constraint_comps
        self.as_penalty = as_penalty

    def setup(self):
        for comp in self.constraint_comps:
            self.add_input('penalty_' + comp.const_id, val=0.0)
        self.add_output('penalty', val=0.0)

    def compute(self, inputs, outputs):
        penalties = np.array([inputs[x] for x in inputs])
        if self.as_penalty:
            penalty = sum(penalties)
        else:
            penalty = 0.0
        outputs['penalty'] = penalty


class PostPenaltyComponent(ExplicitComponent):
    def __init__(self, constraint_comps, as_penalty):
        super().__init__()
        self.constraint_comps = constraint_comps
        self.as_penalty = as_penalty

    def setup(self):
        for comp in self.constraint_comps:
            self.add_input('post_penalty_' + comp[0], val=0.0)
        self.add_output('post_penalty', val=0.0)

    def compute(self, inputs, outputs):
        penalties = np.array([inputs[x] for x in inputs])
        if self.as_penalty:
            penalty = sum(penalties)
        else:
            penalty = 0.0
        outputs['post_penalty'] = penalty
