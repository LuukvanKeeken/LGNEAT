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
    return jnp.asarray(jnp.argmax(z), dtype=jnp.float32)

def argmax_filtered_(z):
        """
        Return the relative index (0.0 or 1.0 as float32) of the maximum among the
        two non-NaN values in `z`.

        Behavior:
        - `z` may contain many NaNs but exactly two non-NaN entries (pre-synaptic
            nodes for this use-case). We consider the first non-NaN encountered as
            position 0 and the second as position 1.
        - If the first non-NaN >= second non-NaN, return 0.0, otherwise 1.0.
        - If both are NaN (unexpected), returns 0.0 (consistent tie-break to first).
        """
        mask = ~jnp.isnan(z)
        idxs = jnp.nonzero(mask, size=2, fill_value=0)[0]
        vals = z[idxs]
        rel_idx = jnp.argmax(vals, axis=0)
        return jnp.asarray(rel_idx, dtype=jnp.float32)


def new_aggregation_func_(z):
    z = jnp.where(jnp.isnan(z), -jnp.inf, z)
    return jnp.asarray(jnp.argmax(z), dtype=jnp.float32)