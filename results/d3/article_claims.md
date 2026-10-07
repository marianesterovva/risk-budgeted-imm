# D3 frozen article claims

## Confirmed

- D3C risk=2.1642, crossed CI [1.5139, 2.8926].
- D3C retained 74.06% of historical K0 filled notional.
- D3C−D3B risk=+0.0279, crossed CI [-0.0876, 0.2178].
- D3C−D3A risk=+0.4282, crossed CI [-0.3948, 1.1826].
- Residual cap fraction=0.0000; selective alarm-positive fraction=0.1691.
- No checkpoint reselection, controller retuning, test access or OOS access was performed.

## Unsupported

- Independent out-of-sample superiority: the seven windows are development validation.
- Guaranteed budget compliance outside the frozen evaluation design.
- Statistical order-aware superiority unless the D3C−D3B upper CI is below zero.
- Live-trading validity or execution-model robustness.

## Final route

`FREEZE_NEGATIVE_D3_RESULT_NO_OOS_AUTHORIZATION`
