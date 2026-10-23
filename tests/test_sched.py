from flowers.training.optim import cosine_with_warmup

def test_sched():
    assert cosine_with_warmup(0, 100, 10) < 0.2
    assert abs(cosine_with_warmup(10, 100, 10) - 1.0) < 1e-6
    assert cosine_with_warmup(99, 100, 10) < 0.01
