# Frozen article claims

## Confirmed

- K4 reduced mean risk from 3.617 to 1.270 (64.88% relative reduction).
- D2 retained 66.18% of K0 filled notional and passed the 50% activity floor.
- D2 did not satisfy the absolute loose budget: risk 2.272 versus budget 1.214; crossed CI [1.346, 3.574].
- D2 was worse than K4 in crossed inference: Δrisk=+1.001, CI [0.164, 2.359].
- The frozen v0.9 decision is REPORT_NEGATIVE_RESULT_NO_OOS_AUTHORIZATION; the test split remains untouched.

## Suggestive but not confirmed

- D2 reduced point-estimate risk by 56.29% relative to D1, but Δrisk CI [-7.171, 0.494] crossed zero.
- D2 reduced point-estimate risk by 37.50% relative to D0, with a near-boundary crossed CI [-3.630, 0.047].
- D2 mean PnL was 2.846; its improvement over K4 was not established because CI [-9.545, 28.854] crossed zero.
- The post-hoc controller diagnostics should be used to discuss multiplier saturation and alarm calibration, not to redefine the primary success criterion.

## Unsupported and prohibited

- D2 guarantees or empirically establishes absolute budget compliance.
- The order-aware component is statistically superior to realized-only pacing.
- D2 produces a statistically established PnL improvement.
- The method has been validated out of sample or in live trading.

## Recommended framing

The study shows that strong relative inventory-risk reduction does not automatically imply absolute risk-budget compliance. The order-aware controller preserves economically meaningful activity and improves point-estimate risk relative to realized-only pacing, but the incremental effect is heterogeneous across seeds and the dual multiplier exhibits saturation. The contribution is therefore a controlled comparative study and a diagnosis of a concrete failure mode, not a claim of solved budget compliance.
