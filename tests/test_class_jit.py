from typing import NamedTuple
import jax
from functools import partial
import os
import jax.numpy as jnp
from jax.config import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "DEBUG"
config.update("jax_log_compiles", 1)


def jit_instancemethod_recompile_test():
    jax.log_compiles(True)

    class Foo:
        def __init__(self, b):
            self.a = 1
            self.b = b
            self.c = lambda x: x * 2
            self.update = jax.jit(self.update)

        def update(self, x):
            print("Compiles")
            return self.c(x) / self.b

    foo = Foo(5)
    x = jax.numpy.array((2, 3))

    # should compile with correct results
    print("First run")
    y_hat = foo.update(x)
    y = x * 2 / 5
    assert jax.numpy.array_equal(y_hat, y)
    print(y)

    # should not recompile with correct results
    foo.a = 5
    print("Changing a")
    y_hat = foo.update(x)
    y = x * 2 / 5
    assert jax.numpy.array_equal(y_hat, y)
    print(y)

    # should not recompile, but results are wrong
    foo.b = 4
    print("Changing b")
    y_hat = foo.update(x)
    y = x * 2 / 4
    assert not jax.numpy.array_equal(y_hat, y)
    print(y)

    # should recompile with correct results
    foo = Foo(9)
    print("Reinstantiating foo")
    y_hat = foo.update(x)
    y = x * 2 / 9
    assert jax.numpy.array_equal(y, x * 2 / 9)
    print(y)

    for i in range(100):
        k = jax.random.PRNGKey(i)
        x = jax.random.normal(k, (64, 64))
        foo.update(x)
        print(i)

    return


def hparams_mutability_after_jit_test():
    class Properties(NamedTuple):
        a: int
        b: int

    class Foo:
        def __init__(self, properties: Properties):
            self.properties = properties

        @partial(jax.jit, static_argnums=0)
        def update(self, x):
            return self.properties.a / self.properties.b * x

    class Bar(Foo):
        def __init__(self, properties):
            super().__init__(properties)
            self.properties = self.properties._replace(a=3)

    class BarJ(Foo):
        def __init__(self, properties):
            super().__init__(properties)
            self.properties._replace(a=3)
            self.update = jax.jit(self.update.__wrapped__, static_argnums=0)

    x = jax.numpy.array((2, 3))
    properties = Properties(1, 2)

    foo = Foo(properties)
    y = foo.update(x)
    assert jax.numpy.array_equal(y, 1 / 2 * x)
    print(y)

    bar = Bar(properties)
    y = bar.update(x)
    assert jax.numpy.array_equal(y, 3 / 2 * x)
    print(y)

    bar_j = BarJ(properties)
    y = bar_j.update(x)
    assert jax.numpy.array_equal(y, 3 / 2 * x)
    print(y)


def test_print_jitted():
    def loss(x, y, z):
        print("Compiles")
        a = x * y
        b = jnp.log(z) * a
        return b

    f = jax.jit(loss)
    for i in range(100):
        s = (64, 64)
        k = jax.random.PRNGKey(i)
        x = jax.random.normal(k, s)
        y = jax.random.normal(k, s)
        z = jax.random.normal(k, s)
        r = f(x, y, z)
        r_2 = r * 10
        print(i, r_2.mean())


def test_print_jitted_tuple_input():
    def loss(x, Y, z):
        print("Compiles")
        y_1, y_2, y_3 = Y
        a = (x * y_1 / y_2 * y_3)
        b = jnp.log(z) * a
        return b

    f = jax.jit(loss)
    for i in range(100):
        s = (64, 64)
        k = jax.random.PRNGKey(i)
        x = jax.random.normal(k, s)
        y = jax.random.normal(k, s)
        z = jax.random.normal(k, s)
        Y = (y, y + 10, y / 20)
        r = f(x, Y, z)
        r_2 = r * 10
        print(i, r_2.mean())


if __name__ == "__main__":
    # jit_instancemethod_recompile_test()
    # hparams_mutability_after_jit_test()
    # test_print_jitted()
    test_print_jitted_tuple_input()