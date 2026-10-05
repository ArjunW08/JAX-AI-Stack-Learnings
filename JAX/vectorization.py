

import jax
import jax.numpy as jnp
import time
import numpy as np

key = jax.random.key(42)

W = jax.random.normal(key, (150,100)) # 100 values per input sample, 150 neurons in next layer
X = jax.random.normal(key, (10, 100))

def calculate_output(X):
    return jnp.dot(W, X)

def batched_calculation_loop(X):
    return jnp.stack([calculate_output(x) for x in X])

def batched_calculation_manual(X):
    return jnp.dot(X, W.T)

batched_calculation_vmap = jax.vmap(calculate_output)

start = time.perf_counter()
batched_calculation_loop(X)
end = time.perf_counter()
print(end - start)

start = time.perf_counter()
batched_calculation_manual(X)
end = time.perf_counter()
print(end - start)

start = time.perf_counter()
batched_calculation_vmap(X)
end = time.perf_counter()
print(end - start)

np.testing.assert_allclose(batched_calculation_loop(X), batched_calculation_manual(X), atol=1E-4, rtol=1E-4)
np.testing.assert_allclose(batched_calculation_manual(X), batched_calculation_vmap(X), atol=1E-4, rtol=1E-4)