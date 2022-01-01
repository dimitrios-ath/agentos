import agentos


def test_component_run():
    class Simple:
        def fn(self, input):
            return input

    params = agentos.ParameterSet({"Simple": {"fn": {"input": "hi"}}})
    c = agentos.Component.from_class(Simple)
    r = c.run("fn", params)
    print(r.identifier)