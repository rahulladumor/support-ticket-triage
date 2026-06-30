from ticket_triage.leakage import group_labels, template_skeleton


def test_template_variants_collapse_to_same_skeleton():
    # Same template, different coin / prefix / suffix / amount -> one group.
    a = "How does staking work and what rewards can I expect on Polygon? Please advise."
    b = "Hey, How does staking work and what rewards can I expect on Ethereum? Thanks."
    assert template_skeleton(a) == template_skeleton(b)

    c = "Urgent: Someone withdrew $50 of BTC from my account that I never authorized."
    d = "Someone withdrew $10,000 of Bitcoin from my account that I never authorized. This is time sensitive."
    assert template_skeleton(c) == template_skeleton(d)


def test_distinct_intents_have_distinct_skeletons():
    staking = "How does staking work and what rewards can I expect on SOL?"
    minimum = "What's the minimum amount I can buy of Cardano?"
    assert template_skeleton(staking) != template_skeleton(minimum)


def test_group_labels_reduce_templated_rows():
    # A batch of coin-swapped duplicates collapses to far fewer groups than rows.
    rows = [f"How long do {c} withdrawals usually take to process?" for c in ("BTC", "ETH", "SOL", "XRP")]
    assert len(set(group_labels(rows))) == 1
