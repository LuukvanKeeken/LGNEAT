import jax.numpy as jnp


def sum_(z):
    return jnp.sum(z, axis=0, where=~jnp.isnan(z), initial=0)


def product_(z):
    return jnp.prod(z, axis=0, where=~jnp.isnan(z), initial=1)


def max_(z):
    return jnp.max(z, axis=0, where=~jnp.isnan(z), initial=-jnp.inf)


def min_(z):
    return jnp.min(z, axis=0, where=~jnp.isnan(z), initial=jnp.inf)


def maxabs_(z):
    z = jnp.where(jnp.isnan(z), 0, z)
    abs_z = jnp.abs(z)
    max_abs_index = jnp.argmax(abs_z)
    return z[max_abs_index]

def mean_(z):
    sumation = sum_(z)
    valid_count = jnp.sum(~jnp.isnan(z), axis=0)
    return sumation / valid_count

# Return an array containing only the non-NaN values from z.
# There will always be exactly 2 non-NaN values.
def filter_nans_(z):
    
    mask = ~jnp.isnan(z)
    idxs = jnp.nonzero(mask, size=2, fill_value=0)[0]
    return z[idxs]

# Also check if z is an array of two elements
def argmax_(z):
    """
    Return the index (0 or 1, as a float) of the maximum element along the
    last axis for 2-element inputs. NaNs are treated as -inf so they never win.

    The result is a float (0.0 or 1.0) which matches downstream code that
    expects numeric labels/activations rather than integer indices.
    """
    # make NaNs effectively -inf
    z = jnp.where(jnp.isnan(z), -jnp.inf, z)
    # use jnp.argmax which is jittable and returns the first index on ties
    idx = jnp.argmax(z, axis=0)
    return jnp.asarray(idx, dtype=jnp.float32)