from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import jax.random


class MCMC(ABC):
    """
    This class wraps the the Metropolis Hasting Sampler.
    """
    def __init__(self, shape, logprob):
        self.shape = shape
        self.logprob = logprob

    @abstractmethod
    def propose(self, key, element):
        """
        Proposes a new Sample.
        """
        pass

    def next_element(self, key, element):
        """
        Performs a Metropolis step. Returns new element and boolean value if proposed sample was accepted.
        """
        key, subkey = jax.random.split(key)

        subsubkey, proposal = self.propose(key, element)

        ratio = jnp.exp(self.logprob(proposal) - self.logprob(element))

        return jnp.where(jax.random.uniform(subkey) < ratio, proposal,  jnp.copy(element)), jnp.where(jax.random.uniform(subkey) < ratio, 1, 0)
    

    @partial(jax.jit, static_argnames=['self', 'N_samples'])
    def sample(self, rkey, initial, N_samples):
        """
        Samples a markov chain of length N_samples
        """
        
        def f(data, bob):
            next_item, accept = self.next_element(data[0], data[1])
            key, _ = jax.random.split(data[0])
            return ((key, next_item, data[2] + accept), next_item)
        
        carry, samples = jax.lax.scan(f, (rkey, initial, 0), jnp.zeros((N_samples,) + self.shape))

        return samples, carry[2]/N_samples