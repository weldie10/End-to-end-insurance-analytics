"""Statistical testing and A/B risk hypothesis testing modules."""

from .ab_risk_hypothesis_tester import ABRiskHypothesisTester

# Backward compatibility alias
HypothesisTester = ABRiskHypothesisTester

__all__ = ["ABRiskHypothesisTester", "HypothesisTester"]

