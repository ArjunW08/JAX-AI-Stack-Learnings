import jax
import jax.numpy as jnp

def square(x):
    return x ** 2

'''
x = 10
f(x) = x ^ 2
f'(x) = 2x
f''(x) = 2
f''\'(x) = 0
'''

# Differentiation is done using grad function

value = 10.0
print(f"Value : {value}")
print(f"Square to : {square(value)}")

print(f"First Derivative : {jax.grad(square, allow_int=True)(value)}")
print(f"Second Derivative : {jax.grad(jax.grad(square), allow_int=True)(value)}")
print(f"Third Derivative : {jax.grad(jax.grad(jax.grad(square)), allow_int=True)(value)}")

# Partial Derivatives

def f(x, y, z):
    return x ** 2 + 2 * y ** 2 + 3 * z ** 2

x, y, z = 2.0, 4.0, 6.0


print(f(x,y,z))
print(jax.grad(f, argnums=0, allow_int=True)(x,y,z))
print(jax.grad(f, argnums=1, allow_int=True)(x,y,z))
print(jax.grad(f, argnums=2, allow_int=True)(x,y,z))

arr = [2.0,4.0,6.0]

def f2(arr):
    return arr[0] ** 2 + 2 * arr[1] ** 2 + 3 * arr[2] ** 2

print(f2(arr))
print(jax.grad(f2, allow_int=True)(arr))
