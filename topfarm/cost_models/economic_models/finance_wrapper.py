from topfarm.cost_models.cost_model_wrappers import CostModelComponent
import numpy as np
from costmodels.finance import Product, Technology
from costmodels.models import DTUOffshoreCostModel
from costmodels.project import Project


class OffshoreWindPlantFinanceWrapper(CostModelComponent):
    def __init__(self, windTurbines, n_wt, el_price, LIFETIME=25, **kwargs):
        cost_model = DTUOffshoreCostModel(
            # water_depth=30.0, # DYNAMIC
            # aep=1.0e9, # DYNAMIC
            rated_power=windTurbines.power(20) / 1e6,
            rotor_speed=10.0,
            rotor_diameter=windTurbines.diameter(),
            hub_height=windTurbines.hub_height(),
            lifetime=LIFETIME,
            capacity_factor=0.4,
            nwt=n_wt,
        )
        self.cost_model = cost_model

        # out = cost_model.run(aep=373206.64521435613, water_depth=20.0)
        # print(out)

        wind_technology = Technology(
            name="wind",
            lifetime=LIFETIME,
            product=Product.SPOT_ELECTRICITY,
            cost_model=cost_model,
        )

        wind_farm_project = Project(
            technologies=[wind_technology],
            product_prices={Product.SPOT_ELECTRICITY: el_price},  # np.random.uniform(0, el_price, LIFETIME)
        )

        def npv_func(aep, water_depth, **kwargs):
            return np.asarray(
                wind_farm_project.npv(
                    productions={
                        wind_technology.name: aep,
                    },
                    cost_model_args={
                        wind_technology.name: {
                            "water_depth": water_depth,
                            "aep": aep,
                        }
                    },
                )
            )

        def npv_grad_func(aep, water_depth, **kwargs):
            grads = wind_farm_project.npv_grad(
                productions={
                    wind_technology.name: aep,
                },
                cost_model_args={
                    wind_technology.name: {
                        "water_depth": water_depth,
                        "aep": aep,
                    }
                },
            )
            prod_grad = grads[0][wind_technology.name]
            water_depth_grad = grads[1][wind_technology.name]["water_depth"]
            return np.asarray(prod_grad), np.asarray(water_depth_grad)

        CostModelComponent.__init__(self,
                                    input_keys=[
                                        ("aep", 0),
                                        ("water_depth", 30 * np.ones(n_wt)),
                                    ],
                                    n_wt=n_wt,
                                    cost_function=npv_func,
                                    cost_gradient_function=npv_grad_func,
                                    objective=True,
                                    maximize=True,
                                    output_keys=[("npv", 0)],
                                    output_unit="EUR",
                                    )
