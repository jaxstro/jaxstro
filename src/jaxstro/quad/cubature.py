"""Static declarations for multidimensional cubature rules."""

from dataclasses import dataclass

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GenzMalik:
    """Degree-7 Genz-Malik rule with an embedded degree-5 formula."""

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, _metadata, _children):
        return cls()
