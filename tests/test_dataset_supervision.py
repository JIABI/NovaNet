from novanet.config import load_config
from novanet.dataset import _hof_supervision
from novanet.handover import CHOOutcome


def test_counterfactual_hof_target_is_excluded_from_head_bce():
    config = load_config()
    not_attempted = CHOOutcome(
        attempted=False,
        success=False,
        failure_reason="ttt_or_hysteresis_not_sustained",
        completion_time_s=None,
        interruption_s=0.0,
    )
    target, mask = _hof_supervision(
        not_attempted,
        lambda _offset_s: -20.0,
        config,
    )
    assert target == 1.0  # retained for the all-pair clairvoyant teacher
    assert mask is False  # excluded from the attempted-execution BCE


def test_attempted_hof_label_is_included_in_head_bce():
    config = load_config()
    failed_attempt = CHOOutcome(
        attempted=True,
        success=False,
        failure_reason="target_outage_during_execution",
        completion_time_s=config.handover.ttt_s + config.handover.execution_s,
        interruption_s=config.handover.execution_s,
    )
    target, mask = _hof_supervision(
        failed_attempt,
        lambda _offset_s: 20.0,
        config,
    )
    assert target == 1.0
    assert mask is True
