from mmrl.state.depth_aware_reference_price import (
    DepthAwareStableReferencePriceTracker,
)


def book(bids, asks):
    return (
        {int(p): float(q) for p, q in bids},
        {int(p): float(q) for p, q in asks},
    )


def test_known_empty_uses_paper_path():
    t = DepthAwareStableReferencePriceTracker(initial_even_tie_break="lower")
    b, a = book(
        [(100,1),(99,1),(98,1),(97,1),(96,1)],
        [(102,1),(103,1),(104,1),(105,1),(106,1)],
    )
    assert t.reset(b,a).ref2 == 201
    b, a = book(
        [(101,1),(99,1),(98,1),(97,1),(96,1)],
        [(103,1),(104,1),(105,1),(106,1),(107,1)],
    )
    s = t.update(b,a)
    assert s.paper_gate_empty
    assert not s.depth_truncation_reanchor


def test_nonempty_gate_holds():
    t = DepthAwareStableReferencePriceTracker(initial_even_tie_break="lower")
    b, a = book(
        [(100,1),(99,1),(98,1),(97,1),(96,1)],
        [(102,1),(103,1),(104,1),(105,1),(106,1)],
    )
    t.reset(b,a)
    old = t.ref2
    b, a = book(
        [(100,1),(99,1),(98,1),(97,1),(96,1)],
        [(103,1),(104,1),(105,1),(106,1),(107,1)],
    )
    s = t.update(b,a)
    assert s.ref2 == old
    assert not s.depth_truncation_reanchor


def test_unknown_mid_up_reanchors():
    t = DepthAwareStableReferencePriceTracker(initial_even_tie_break="lower")
    b, a = book(
        [(100,1),(99,1),(98,1),(97,1),(96,1)],
        [(102,1),(103,1),(104,1),(105,1),(106,1)],
    )
    t.reset(b,a)
    t.ref2 = 189
    t.prev_mid2 = 202
    b, a = book(
        [(100,1),(99,1),(98,1),(97,1),(96,1)],
        [(103,1),(104,1),(105,1),(106,1),(107,1)],
    )
    s = t.update(b,a)
    assert s.depth_truncation_reanchor
    assert s.gate_known is False
    assert s.ref2 == s.candidate_ref2


def test_unknown_mid_down_reanchors():
    t = DepthAwareStableReferencePriceTracker(initial_even_tie_break="upper")
    b, a = book(
        [(100,1),(99,1),(98,1),(97,1),(96,1)],
        [(102,1),(103,1),(104,1),(105,1),(106,1)],
    )
    t.reset(b,a)
    t.ref2 = 219
    t.prev_mid2 = 202
    b, a = book(
        [(99,1),(98,1),(97,1),(96,1),(95,1)],
        [(101,1),(102,1),(103,1),(104,1),(105,1)],
    )
    s = t.update(b,a)
    assert s.depth_truncation_reanchor
    assert s.gate_known is False
    assert s.ref2 == s.candidate_ref2
