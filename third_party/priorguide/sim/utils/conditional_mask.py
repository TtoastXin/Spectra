import jax
import jax.numpy as jnp
import jax.random as jrandom

from functools import partial


def sample_random_conditional_mask(
    key, num_samples, theta_dim, x_dim, alpha=1.0, beta=4.0
):
    # More likely to condition on a few nodes
    key1, key2 = jax.random.split(key, 2)
    condition_mask = jax.random.bernoulli(
        key1,
        jax.random.beta(key2, alpha, beta, shape=(num_samples, 1)),
        shape=(num_samples, theta_dim + x_dim),
    ).astype(jnp.bool_)
    all_ones_mask = jnp.all(condition_mask, axis=-1)
    # If all are ones, then set to false
    condition_mask = jnp.where(all_ones_mask[..., None], False, condition_mask)
    return condition_mask


def joint_conditional_mask(key, num_samples, theta_dim, x_dim):
    return jnp.array([[False] * (theta_dim + x_dim)]*num_samples)


def posterior_conditional_mask(key, num_samples, theta_dim, x_dim):
    return jnp.array([[False] * theta_dim + [True] * x_dim]*num_samples)


def likelihood_conditional_mask(key, num_samples, theta_dim, x_dim):
    return jnp.array([[True] * theta_dim + [False] * x_dim]*num_samples)


def partial_data_conditional_mask(key, num_samples, theta_dim, x_dim):
    n_unknown = int(jnp.ceil(x_dim / 2))
    n_known = x_dim - n_unknown
    mask = jnp.array([[False] * theta_dim + [True] * x_dim] * num_samples)
    # Split the key into num_samples keys
    keys = jax.random.split(key, num_samples)
    for i in range(num_samples):
        x_indices = jax.random.choice(keys[i], x_dim, shape=(n_unknown,), replace=False)
        mask = mask.at[i, theta_dim + x_indices].set(False)
    return mask


def partial_half_data_conditional_mask(key, num_samples, theta_dim, x_dim):

    mask = jnp.array([[False] * theta_dim + [True] * x_dim] * num_samples)
    n_unknown = int(jnp.ceil(x_dim / 2))

    first_half = jax.random.bernoulli(key)  # returns True or False with 50% chance

    if first_half:
        mask = mask.at[:, theta_dim:theta_dim+n_unknown].set(False)
    else:
        mask = mask.at[:, theta_dim+n_unknown-1:theta_dim+x_dim].set(False)

    return mask

def all_false_conditional_mask(key, num_samples, theta_dim, x_dim):
    return jnp.array([[False] * (theta_dim + x_dim)] * num_samples)


def sample_strutured_conditional_mask(
    key,
    num_samples,
    theta_dim,
    x_dim,
    p_joint=0.2,
    p_posterior=0.2,
    p_likelihood=0.2,
    p_rnd1=0.2,
    p_rnd2=0.2,
    rnd1_prob=0.3,
    rnd2_prob=0.7,
):
    # Joint, posterior, likelihood, random1_mask, random2_mask
    key1, key2, key3 = jax.random.split(key, 3)
    condition_mask = jax.random.choice(
        key1,
        jnp.array(
            [[False] * (theta_dim + x_dim)]
            + [[False] * theta_dim + [True] * x_dim]
            + [
                [True] * theta_dim + [False] * x_dim,
                jax.random.bernoulli(
                    key2, rnd1_prob, shape=(theta_dim + x_dim,)
                ).astype(jnp.bool_),
                jax.random.bernoulli(
                    key3, rnd2_prob, shape=(theta_dim + x_dim,)
                ).astype(jnp.bool_),
            ]
        ),
        shape=(num_samples,),
        p=jnp.array([p_joint, p_posterior, p_likelihood, p_rnd1, p_rnd2]),
        axis=0,
    )
    all_ones_mask = jnp.all(condition_mask, axis=-1)
    # If all are ones, then set to false
    condition_mask = jnp.where(all_ones_mask[..., None], False, condition_mask)
    return condition_mask


def get_condition_mask_fn(name, **kwargs):
    # Simple functions that don't need any kwargs
    if name.lower() == "joint":
        return joint_conditional_mask
    elif name.lower() == "posterior":
        return posterior_conditional_mask
    elif name.lower() == "likelihood":
        return likelihood_conditional_mask
    elif name.lower() == "all_false":
        return all_false_conditional_mask
    # Functions that might need kwargs
    elif name.lower() == "structured_random":
        return partial(sample_strutured_conditional_mask, **kwargs)
    elif name.lower() == "random":
        return partial(sample_random_conditional_mask, **kwargs)
    elif name.lower() == "partial_data":
        return partial(partial_half_data_conditional_mask, **kwargs)
    elif name.lower() == "partial_data_scrambled":
        return partial(partial_data_conditional_mask, **kwargs)
    else:
        raise NotImplementedError()