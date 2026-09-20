import jax
import jax.numpy as jnp

arr = jnp.array([2,3,4])

print(arr)
print(jnp.square(arr))
print(jnp.reshape(arr,(3,1)))