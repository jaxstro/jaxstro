# Registry-provenance fixture

A minimal registry root that loads cleanly through `jaxstro.registry.load_registry`:
three source bundles (two citable, one derivation bundle without bibliographic fields),
one shared symbol, and one row or more in each registry-level table. Every id in it
resolves. `tests/unit/test_registry_tables.py` copies it and breaks one thing per test.
