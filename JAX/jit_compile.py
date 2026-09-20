# JIT Compilation in JAX is in asynchronous mode

import time
import jax
import jax.numpy as jnp

@jax.jit
def function(x):
    return jnp.where(x > 0, 2 * x, x * x)

arr = jnp.arange(0,10,2)
_ = function(arr) #warm_up
start = time.perf_counter()
function(arr).block_until_ready()
end = time.perf_counter()

print(f"Time : {end-start} seconds")

# Byte representation of compiled function
print(jax.make_jaxpr(function)(arr))

# Certain cases where yo cannot use JIT compilation

# @jax.jit
def func(x):
    # return jnp.where(x % 2 == 0, 1, 0) #this will work
    if x % 2 == 0: #this will not work as it is a python control flow and not a jax control flow
        return 1
    else : 
        return 0

func(10)