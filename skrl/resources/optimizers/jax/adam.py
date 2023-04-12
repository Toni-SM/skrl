import flax
import optax
import jax.numpy as jnp

from skrl.models.jax import Model


class Adam:
    def __new__(cls, model: Model, lr: float = 1e-3) -> "Optimizer":
        """Adam optimizer

        :param model: Model
        :type model: skrl.models.jax.Model
        :param lr: Learning rate (default: 1e-3)
        :type lr: float, optional

        :return: Adam optimizer
        :rtype: flax.struct.PyTreeNode
        """
        class Optimizer(flax.struct.PyTreeNode):
            """Optimizer

            This class is the result of isolating the Optax optimizer,
            which is mixed with the model parameters, from flax's TrainState class

            https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#train-state
            """
            transformation: optax.GradientTransformation = flax.struct.field(pytree_node=False)
            state: optax.OptState = flax.struct.field(pytree_node=True)

            @classmethod
            def _create(cls, *, transformation, state, **kwargs):
                return cls(transformation=transformation, state=state, **kwargs)

            def step(self, grad: jnp.ndarray, model: Model) -> "Optimizer":
                """Performs a single optimization step

                :param grad: Gradients
                :type grad: jnp.ndarray
                :param model: Model
                :type model: skrl.models.jax.Model

                :return: Optimizer
                :rtype: flax.struct.PyTreeNode
                """
                params, optimizer_state = self.transformation.update(grad, self.state, model.state_dict.params)
                params = optax.apply_updates(model.state_dict.params, params)
                model.state_dict = model.state_dict.replace(params=params)
                return self.replace(state=optimizer_state)

        transformation = optax.adam(learning_rate=lr)
        return Optimizer._create(transformation=transformation, state=transformation.init(model.state_dict.params))
