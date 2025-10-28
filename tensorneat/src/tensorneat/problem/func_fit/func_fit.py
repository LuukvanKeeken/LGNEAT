import jax
from jax import vmap, numpy as jnp
import jax.numpy as jnp

from ..base import BaseProblem
from tensorneat.src.tensorneat.common import State


class FuncFit(BaseProblem):
    jitable = True

    def __init__(self, error_method: str = "binary_cross_entropy"):
        super().__init__()

        assert error_method in {"mse", "rmse", "mae", "mape", "binary_cross_entropy"}
        self.error_method = error_method

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state, randkey, act_func, params):

        predict = vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.inputs
        )

        # temp = 0.01
        # predict_exp = jnp.exp(predict / temp)
        # predict_prob = predict_exp / jnp.sum(predict_exp, axis=1, keepdims=True)

        if self.error_method == "mse":
            loss = jnp.mean((predict - self.targets) ** 2)
            # loss = jnp.mean((predict_prob - self.targets) ** 2)

        elif self.error_method == "rmse":
            loss = jnp.sqrt(jnp.mean((predict - self.targets) ** 2))

        elif self.error_method == "mae":
            loss = jnp.mean(jnp.abs(predict - self.targets))

        elif self.error_method == "mape":
            loss = jnp.mean(jnp.abs((predict - self.targets) / self.targets))

        elif self.error_method == "binary_cross_entropy":
            epsilon = 1e-7  # small constant to avoid log(0)
            loss = -jnp.mean(
                self.targets * jnp.log(predict + epsilon)
                + (1 - self.targets) * jnp.log(1 - predict + epsilon)
            )
        else:
            raise NotImplementedError

        return -loss
    
    def evaluate_threshold(self, state, randkey, act_func, params, threshold=0.5):
        predict = vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.inputs
        )
        predict = (predict >= threshold).astype(jnp.float32)

        accuracy = jnp.mean(predict == self.targets)
        return accuracy
        

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        predict = vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.inputs
        )
        inputs, target, predict = jax.device_get([self.inputs, self.targets, predict])

        # Binarize predictions
        predict = (predict >= 0.5).astype(jnp.float32)

        fitness = self.evaluate(state, randkey, act_func, params)
        accuracy = self.evaluate_threshold(state, randkey, act_func, params)

        loss = -fitness

        msg = ""
        for i in range(inputs.shape[0]):
            msg += f"input: {inputs[i]}, target: {target[i]}, predict: {predict[i]}\n"
        msg += f"loss: {loss}\n"
        msg += f"accuracy: {accuracy}\n"
        print(msg)

    @property
    def inputs(self):
        raise NotImplementedError

    @property
    def targets(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError
