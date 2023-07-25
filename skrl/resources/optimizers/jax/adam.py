from typing import Optional

import functools

import flax
import jax
import optax

from skrl.models.jax import Model


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@functools.partial(jax.jit, static_argnames=("transformation"))
def _step(transformation, grad, state, state_dict):
    # optax transform
    params, optimizer_state = transformation.update(grad, state, state_dict.params)
    # apply transformation
    params = optax.apply_updates(state_dict.params, params)
    return optimizer_state, state_dict.replace(params=params)


@functools.partial(jax.jit, static_argnames=("transformation"))
def _step_with_scale(transformation, grad, state, state_dict, scale):
    # optax transform
    params, optimizer_state = transformation.update(grad, state, state_dict.params)
    # custom scale
    # https://optax.readthedocs.io/en/latest/api.html?#optax.scale
    params = jax.tree_util.tree_map(lambda params: scale * params, params)
    # apply transformation
    params = optax.apply_updates(state_dict.params, params)
    return optimizer_state, state_dict.replace(params=params)


class Adam:
    def __new__(cls, model: Model, lr: float = 1e-3, grad_norm_clip: float = 0, scale: bool = True) -> "Optimizer":
        """Adam optimizer

        Adapted from `Optax's Adam <https://optax.readthedocs.io/en/latest/api.html?#adam>`_
        to support custom scale (learning rate)

        :param model: Model
        :type model: skrl.models.jax.Model
        :param lr: Learning rate (default: ``1e-3``)
        :type lr: float, optional
        :param grad_norm_clip: Clipping coefficient for the norm of the gradients (default: ``0``).
                               Disabled if less than or equal to zero
        :type grad_norm_clip: float, optional
        :param scale: Whether to instantiate the optimizer as-is or remove the scaling step (default: ``True``).
                      Remove the scaling step if a custom learning rate is to be applied during optimization steps
        :type scale: bool, optional

        :return: Adam optimizer
        :rtype: flax.struct.PyTreeNode

        Example::

            >>> optimizer = Adam(model=policy, lr=5e-4)
            >>> # step the optimizer given a computed gradiend (grad)
            >>> optimizer = optimizer.step(grad, policy)

            # apply custom learning rate during optimization steps
            >>> optimizer = Adam(model=policy, lr=5e-4, scale=False)
            >>> # step the optimizer given a computed gradiend and an updated learning rate (lr)
            >>> optimizer = optimizer.step(grad, policy, lr)
        """
        class Optimizer(flax.struct.PyTreeNode):
            """Optimizer

            This class is the result of isolating the Optax optimizer,
            which is mixed with the model parameters, from Flax's TrainState class

            https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#train-state
            """
            transformation: optax.GradientTransformation = flax.struct.field(pytree_node=False)
            state: optax.OptState = flax.struct.field(pytree_node=True)

            @classmethod
            def _create(cls, *, transformation, state, **kwargs):
                return cls(transformation=transformation, state=state, **kwargs)

            def step(self, grad: jax.Array, model: Model, lr: Optional[float] = None) -> "Optimizer":
                """Performs a single optimization step

                :param grad: Gradients
                :type grad: jax.Array
                :param model: Model
                :type model: skrl.models.jax.Model
                :param lr: Learning rate.
                           If given, a scale optimization step will be performed
                :type lr: float, optional

                :return: Optimizer
                :rtype: flax.struct.PyTreeNode
                """
                if lr is None:
                    optimizer_state, model.state_dict = _step(self.transformation, grad, self.state, model.state_dict)
                else:
                    optimizer_state, model.state_dict = _step_with_scale(self.transformation, grad, self.state, model.state_dict, -lr)
                return self.replace(state=optimizer_state)

        # default optax transformation
        if scale:
            transformation = optax.adam(learning_rate=lr)
        # optax transformation without scaling step
        else:
            transformation = optax.scale_by_adam()

        # clip updates using their global norm
        if grad_norm_clip > 0:
            transformation = optax.chain(optax.clip_by_global_norm(grad_norm_clip), transformation)

        return Optimizer._create(transformation=transformation, state=transformation.init(model.state_dict.params))
