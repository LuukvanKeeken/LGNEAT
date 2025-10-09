import jax

import jax.numpy as jnp

def check_jax_gpu():
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    print(f"JAX devices: {devices}")
    if gpu_devices:
        print("GPU is available for JAX!")
        # Simple computation on GPU
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sin(x)
        print("Computation result on GPU:", y)
    else:
        print("No GPU found for JAX.")

if __name__ == "__main__":
    check_jax_gpu()