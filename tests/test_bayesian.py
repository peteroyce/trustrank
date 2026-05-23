from engine.scoring.bayesian import BetaBinomial, DirichletMultinomial


class TestBetaBinomial:
    def test_prior_only(self):
        bb = BetaBinomial(alpha=2.0, beta=2.0)
        assert bb.mean() == 0.5

    def test_update_with_successes(self):
        bb = BetaBinomial(alpha=2.0, beta=2.0)
        bb.update(successes=8, failures=2)
        assert bb.mean() > 0.5
        assert bb.alpha == 10.0
        assert bb.beta == 4.0

    def test_confidence_increases_with_data(self):
        few = BetaBinomial(alpha=2.0, beta=2.0)
        few.update(successes=3, failures=1)
        many = BetaBinomial(alpha=2.0, beta=2.0)
        many.update(successes=80, failures=20)
        assert many.confidence() > few.confidence()

    def test_variance_decreases_with_data(self):
        few = BetaBinomial(alpha=2.0, beta=2.0)
        few.update(successes=3, failures=1)
        many = BetaBinomial(alpha=2.0, beta=2.0)
        many.update(successes=80, failures=20)
        assert many.variance() < few.variance()


class TestDirichletMultinomial:
    def test_uniform_prior(self):
        dm = DirichletMultinomial(alphas=[1.0, 1.0, 1.0, 1.0, 1.0])
        means = dm.mean_distribution()
        assert all(abs(m - 0.2) < 0.001 for m in means)

    def test_update_with_counts(self):
        dm = DirichletMultinomial(alphas=[1.0, 1.0, 1.0, 1.0, 1.0])
        dm.update(counts=[0, 0, 5, 10, 85])
        means = dm.mean_distribution()
        assert means[4] > means[0]

    def test_weighted_mean_as_score(self):
        dm = DirichletMultinomial(alphas=[1.0, 1.0, 1.0, 1.0, 1.0])
        dm.update(counts=[0, 0, 0, 0, 100])
        score = dm.weighted_mean()
        assert score > 4.5

    def test_bimodal_has_high_variance(self):
        bimodal = DirichletMultinomial(alphas=[1.0, 1.0, 1.0, 1.0, 1.0])
        bimodal.update(counts=[50, 0, 0, 0, 50])
        uniform = DirichletMultinomial(alphas=[1.0, 1.0, 1.0, 1.0, 1.0])
        uniform.update(counts=[0, 0, 100, 0, 0])
        assert bimodal.variance() > uniform.variance()
