from agentos.runoutput import RunOutput


def test_tracker():
    t = RunOutput()
    assert t.identifier == t._mlflow_run.info.run_id
    t.log_metric("test_metric", 1)
    assert t.data.metrics["test_metric"] == 1
    t.set_tag("test_tag", "tag_val")
    assert t.data.tags["test_tag"] == "tag_val"
