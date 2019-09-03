# %%
import topfarm
import openmdao.api as om
from topfarm.cost_models.utils.spanning_tree import spanning_tree
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
import numpy as np

class ElNetComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_wt', types=int)

    # @property
    # def n_wt(self):
    #     return self.options['n_wt']

    def __getattr__(self, key):
        if key in self.options:
            return self.options[key]
        else:
            super().__getattr__(key)

    def setup(self): 
        self.add_input(topfarm.x_key, np.zeros(self.n_wt), units='m')
        self.add_input(topfarm.y_key, np.zeros(self.n_wt), units='m')

#%%
enc = ElNetComp(n_wt=10)

#%%
