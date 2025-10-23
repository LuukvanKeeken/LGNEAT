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
    # Return the non-NaN values (up to two) in their original order.
    # If there are fewer than two non-NaNs, pad the result with -inf so
    # downstream reductions (argmax) don't see NaN and behave consistently
    # across devices and compilation modes.
    mask = ~jnp.isnan(z)
    z_safe = jnp.where(mask, z, -jnp.inf)

    # positions, with NaNs pushed to the end (pos_masked >= len(z) for NaNs)
    pos = jnp.arange(z.shape[0])
    pos_masked = jnp.where(mask, pos, z.shape[0])

    # take the first two positions that were non-NaN (original order)
    top2_pos = jnp.argsort(pos_masked)[:2]
    return z_safe[top2_pos]

# Also check if z is an array of two elements
def argmax_(z):
    return jnp.asarray(jnp.argmax(z), dtype=jnp.float32)