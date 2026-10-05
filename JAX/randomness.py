# Three requirements to fulfill using randomness
# 1. Reproducibilty : Same result given the same seed
# 2. Parallelizability : 
# 3. Vectorizability : 

import jax

key = jax.random.key(42)

keys = jax.random.split(key, 10)

print(keys)
