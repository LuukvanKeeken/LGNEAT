from typing import Callable, Union, List, Tuple
from jax import vmap, Array, numpy as jnp
import numpy as np
import jax

from .func_fit import FuncFit


class CustomFuncFit(FuncFit):

    def __init__(
        self,
        func: Callable,
        low_bounds: Union[List, Tuple, Array],
        upper_bounds: Union[List, Tuple, Array],
        method: str = "sample",
        num_samples: int = 100,
        step_size: Array = None,
        train_test_split: float = 0.8,
        split_seed: int = 42,
        input_data: Array = None,
        output_data: Array = None,
        *args,
        **kwargs,
    ):

        if isinstance(low_bounds, list) or isinstance(low_bounds, tuple):
            low_bounds = np.array(low_bounds, dtype=np.float32)
        if isinstance(upper_bounds, list) or isinstance(upper_bounds, tuple):
            upper_bounds = np.array(upper_bounds, dtype=np.float32)

        assert method in {"sample", "grid", "set_direct"}, "Unknown method"

        if method != "set_direct":
            try:
                out = func(low_bounds)
            except Exception as e:
                raise ValueError(f"func(low_bounds) raise an exception: {e}")
            assert low_bounds.shape == upper_bounds.shape

        

        if method == "set_direct":
            assert input_data is not None and output_data is not None, "input_data and output_data must be provided when method is 'set_direct'"
            self.data_inputs = jnp.array(input_data)
            self.data_outputs = jnp.array(output_data)

        self.func = func
        self.low_bounds = low_bounds
        self.upper_bounds = upper_bounds

        self.method = method
        self.num_samples = num_samples
        self.step_size = step_size

        self.train_test_split = train_test_split
        self.split_seed = split_seed

        self.generate_dataset()

        super().__init__(*args, **kwargs)

    def generate_dataset(self):

        if self.method == "sample":
            assert (
                self.num_samples > 0
            ), f"num_samples must be positive, got {self.num_samples}"

            inputs = np.zeros(
                (self.num_samples, self.low_bounds.shape[0]), dtype=np.float32
            )
            for i in range(self.low_bounds.shape[0]):
                inputs[:, i] = np.random.uniform(
                    low=self.low_bounds[i],
                    high=self.upper_bounds[i],
                    size=(self.num_samples,),
                )

            outputs = vmap(self.func)(inputs)

            self.data_inputs = jnp.array(inputs)
            self.data_outputs = jnp.array(outputs)
        elif self.method == "grid":
            assert (
                self.step_size is not None
            ), "step_size must be provided when method is 'grid'"
            assert (
                self.step_size.shape == self.low_bounds.shape
            ), "step_size must have the same shape as low_bounds"
            assert np.all(self.step_size > 0), "step_size must be positive"

            inputs = np.zeros((1, 1))
            for i in range(self.low_bounds.shape[0]):
                new_col = np.arange(
                    self.low_bounds[i], self.upper_bounds[i], self.step_size[i]
                )
                inputs = cartesian_product(inputs, new_col[:, None])
            inputs = inputs[:, 1:]

            outputs = vmap(self.func)(inputs)

            self.data_inputs = jnp.array(inputs)
            self.data_outputs = jnp.array(outputs)
        elif self.method == "set_direct":
            pass
        else:
            raise ValueError(f"Unknown method: {self.method}")

        

        # Create train/test split
        rng_state = np.random.get_state()
        np.random.seed(self.split_seed)
        num_data = self.data_inputs.shape[0]
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        split_idx = int(num_data * self.train_test_split)
        self.train_indices = indices[:split_idx]
        self.test_indices = indices[split_idx:]
        np.random.set_state(rng_state)

    @property
    def inputs(self):
        return self.data_inputs

    @property
    def targets(self):
        return self.data_outputs

    @property
    def input_shape(self):
        return self.data_inputs.shape

    @property
    def output_shape(self):
        return self.data_outputs.shape
    
    @property
    def train_idx(self):
        return self.train_indices
    
    @property
    def test_idx(self):
        return self.test_indices
    


    def evaluate(self, state, randkey, act_func, params, train = True):

        predict = vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.inputs[self.train_idx if train else self.test_idx]
        )

        targets = self.targets[self.train_idx if train else self.test_idx]

        if self.error_method == "mse":
            loss = jnp.mean((predict - targets) ** 2)

        elif self.error_method == "rmse":
            loss = jnp.sqrt(jnp.mean((predict - targets) ** 2))

        elif self.error_method == "mae":
            loss = jnp.mean(jnp.abs(predict - targets))

        elif self.error_method == "mape":
            loss = jnp.mean(jnp.abs((predict - targets) / targets))

        elif self.error_method == "bce":
            epsilon = 1e-7  # small constant to avoid log(0)
            loss = -jnp.mean(
                targets * jnp.log(predict + epsilon)
                + (1 - targets) * jnp.log(1 - predict + epsilon)
            )

        elif self.error_method == "cce":
            epsilon = 1e-7  # small constant to avoid log(0)
            loss = -jnp.mean(jnp.sum(targets * jnp.log(predict + epsilon), axis=-1))

        else:
            raise NotImplementedError

        return -loss
    

    # def evaluate_threshold(self, state, randkey, act_func, params, threshold=0.5, train = True):
    #     predict = vmap(act_func, in_axes=(None, None, 0))(
    #         state, params, self.inputs[self.train_idx if train else self.test_idx]
    #     )
    #     predict = (predict >= threshold).astype(jnp.float32)

    #     accuracy = jnp.mean(predict == self.targets[self.train_idx if train else self.test_idx])
    #     return accuracy
    

    def evaluate_accuracy(self, state, randkey, act_func, params, train = True):
        predict = vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.inputs[self.train_idx if train else self.test_idx]
        )
        
        # Compare prediction and target by argmax
        predict = jnp.argmax(predict, axis=-1)
        targets = jnp.argmax(self.targets[self.train_idx if train else self.test_idx], axis=-1)

        accuracy = jnp.mean(predict == targets)
        return accuracy



    def show(self, state, randkey, act_func, params, *args, **kwargs):
        predict = vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.inputs
        )
        inputs, target, predict = jax.device_get([self.inputs, self.targets, predict])

        # Turn prediction vectors into one-hot encoded vectors
        predict_bin = jnp.zeros_like(predict)
        predict_indices = jnp.argmax(predict, axis=-1)
        predict_bin = predict_bin.at[jnp.arange(predict.shape[0]), predict_indices].set(1)

        fitness_train = self.evaluate(state, randkey, act_func, params, train=True)
        fitness_test = self.evaluate(state, randkey, act_func, params, train=False)
        accuracy_train = self.evaluate_accuracy(state, randkey, act_func, params, train=True)
        accuracy_test = self.evaluate_accuracy(state, randkey, act_func, params, train=False)

        loss_train = -fitness_train
        loss_test = -fitness_test

        msg = ""
        msg += "Training Data:\n"
        for i in range(inputs[self.train_idx].shape[0]):
            msg += f"target: {target[self.train_idx][i]}, predict: {predict[self.train_idx][i]} {predict_bin[self.train_idx][i]}\n"
            # msg += f"input: {inputs[self.train_idx][i]}, target: {target[self.train_idx][i]}, predict: {predict[self.train_idx][i]} {predict_bin[self.train_idx][i]}\n"
        msg += f"loss: {loss_train}\n"
        msg += f"accuracy: {accuracy_train}\n\n"
        msg += "Testing Data:\n"
        for i in range(inputs[self.test_idx].shape[0]):
            msg += f"target: {target[self.test_idx][i]}, predict: {predict[self.test_idx][i]} {predict_bin[self.test_idx][i]}\n"
            # msg += f"input: {inputs[self.test_idx][i]}, target: {target[self.test_idx][i]}, predict: {predict[self.test_idx][i]} {predict_bin[self.test_idx][i]}\n"
        msg += f"loss: {loss_test}\n"
        msg += f"accuracy: {accuracy_test}\n"
        print(msg)


def cartesian_product(arr1, arr2):
    assert (
        arr1.ndim == arr2.ndim
    ), "arr1 and arr2 must have the same number of dimensions"
    assert arr1.ndim <= 2, "arr1 and arr2 must have at most 2 dimensions"

    len1 = arr1.shape[0]
    len2 = arr2.shape[0]

    repeated_arr1 = np.repeat(arr1, len2, axis=0)
    tiled_arr2 = np.tile(arr2, (len1, 1))

    new_arr = np.concatenate((repeated_arr1, tiled_arr2), axis=1)
    return new_arr
