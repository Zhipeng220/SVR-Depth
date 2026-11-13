"""Candidate protocol checks; a configuration flag is not evidence."""

from svr_depth.evaluation.m01_paper_report import REQUIRED_CHECKS


def audit_status(checks=None):
    checks = checks or {}
    # Each check needs a verified boolean plus a nonempty evidence reference.
    validated = {name: bool(isinstance(checks.get(name), dict)
                            and checks[name].get('passed') is True
                            and isinstance(checks[name].get('evidence'), str)
                            and checks[name]['evidence'].strip()) for name in REQUIRED_CHECKS}
    return {'required_checks': validated, 'missing_checks': [n for n, ok in validated.items() if not ok],
            'formal_aggregate_ready': False,
            'reason': 'Candidate acceptance only; evidence authenticity and author equivalence require independent review.'}
