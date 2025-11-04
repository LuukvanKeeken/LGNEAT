import jax.numpy as jnp

SCALE = 3


def scaled_sigmoid_(z):
    z = 1 / (1 + jnp.exp(-z))
    return z * SCALE


def sigmoid_(z):
    z = 1 / (1 + jnp.exp(-z))
    return z


def scaled_tanh_(z):
    return jnp.tanh(z) * SCALE


def tanh_(z):
    return jnp.tanh(z)


def sin_(z):
    return jnp.sin(z)


def relu_(z):
    return jnp.maximum(z, 0)


def lelu_(z):
    leaky = 0.005
    return jnp.where(z > 0, z, leaky * z)


def identity_(z):
    return z


def inv_(z):
    # avoid division by zero
    z = jnp.where(z > 0, jnp.maximum(z, 1e-7), jnp.minimum(z, -1e-7))
    return 1 / z


def log_(z):
    z = jnp.maximum(z, 1e-7)
    return jnp.log(z)


def exp_(z):
    return jnp.exp(z)


def abs_(z):
    return jnp.abs(z)


def gaussian_(z):
    return jnp.exp(-jnp.square(z))


def cos_(z):
    return jnp.cos(z)


def square_(z):
    return jnp.square(z)


def abs_root_(z):
    return jnp.sqrt(jnp.abs(z))


# NAND gate operation with z having two elements
# Should raise some error if z does not have exactly two elements
def nand_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return 1 - jnp.prod(z)


# NOR gate operation with z having two elements
# Should raise some error if z does not have exactly two elements
def nor_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return 1 - jnp.maximum(z[0], z[1])


# AND gate operation with z having two elements
# Should raise some error if z does not have exactly two elements
def and_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return jnp.prod(z)


# OR gate operation with z having two elements
# Should raise some error if z does not have exactly two elements
def or_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return jnp.maximum(z[0], z[1])


# FALSE gate operation, always returns 0
def false_(z):
    return jnp.array(0.0)


# A AND NOT B gate operation with z having two elements
def a_and_not_b_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return z[0] * (1 - z[1])


# A gate operation, returns the first element
def a_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return z[0]


# XOR gate operation with z having two elements
def xor_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return jnp.where(z[0] != z[1], 1.0, 0.0)


# XNOR gate operation with z having two elements
def xnor_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return jnp.where(z[0] == z[1], 1.0, 0.0)


# NOT A gate operation, returns 1 - first element
def not_a_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return 1 - z[0]


# A OR NOT B gate operation with z having two elements
def a_or_not_b_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return jnp.maximum(z[0], 1 - z[1])


# TRUE gate operation, always returns 1
def true_(z):
    return jnp.array(1.0)


# NOT A AND B gate operation with z having two elements
def not_a_and_b_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return (1 - z[0]) * z[1]


# B gate operation, returns the second element
def b_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return z[1]


# NOT B gate operation, returns 1 - second element
def not_b_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return 1 - z[1]


# NOT A OR B gate operation with z having two elements
def not_a_or_b_(z):
    if z.shape[0] != 2:
        raise ValueError("Input must have exactly two elements")
    return jnp.maximum(1 - z[0], z[1])