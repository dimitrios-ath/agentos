import agentos


def test_component_run():
    class Simple:
        def __init__(self, x):
            self._x = x

        def fn(self, input):
            return self._x, input

    params = agentos.ParameterSet(
        {"Simple": {"__init__": {"x": 1}, "fn": {"input": "hi"}}}
    )
    c = agentos.Component.from_class(Simple)
    r = c.run("fn", params)
    assert r.component == c
    assert r.entry_point == "fn"
