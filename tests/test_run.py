def test_component_run():
    from agentos import ParameterSet, Component

    class Simple:
        def __init__(self, x):
            self._x = x

        def fn(self, input):
            return self._x, input

    params = ParameterSet(
        {"Simple": {"__init__": {"x": 1}, "fn": {"input": "hi"}}}
    )
    c = Component.from_class(Simple)
    r = c.run("fn", params)
    assert r.component == c
    assert r.entry_point == "fn"


def test_run_tracking():
    from agentos import Run
    run = Run()
    assert run.identifier == run._mlflow_run.info.run_id
    run.log_metric("test_metric", 1)
    assert run.data.metrics["test_metric"] == 1
    run.set_tag("test_tag", "tag_val")
    assert run.data.tags["test_tag"] == "tag_val"
