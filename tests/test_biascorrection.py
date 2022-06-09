from functools import partial
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax.config import config
import os


config.update("jax_log_compiles", 1)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


def biascorrection_recompilation_test():
    class M(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(1)(x)
            return x

    m = M()
    o = optax.adam(0.025, b1=int(1))

    input_array = jnp.ones((8,), dtype=float)
    params = m.init(jax.random.PRNGKey(0), input_array)
    opt_state = o.init(params)

    m_forward = jax.vmap(m.apply, in_axes=(None, 0))

    @jax.jit
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params, x, y):
        params_m = params
        y_m = m_forward(params_m, x)
        loss = jnp.mean(y_m - y ** 2)
        return loss, ()



    for i in range(1000):
        print(i)
        inputs = jax.random.normal(jax.random.PRNGKey(i), (32, 8))
        outputs = jax.random.normal(jax.random.PRNGKey(-i), (32, 8))
        _, grads = loss_fn(params, inputs, outputs)
        updates, opt_state = o.update(grads, opt_state)
        params = optax.apply_updates(params, updates)


if __name__ == "__main__":
    biascorrection_recompilation_test()
