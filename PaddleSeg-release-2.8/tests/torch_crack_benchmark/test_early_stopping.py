import unittest


class EarlyStoppingTests(unittest.TestCase):
    def test_early_stopping_stops_after_patience_without_min_delta_improvement(self):
        from torch_crack_benchmark.train import EarlyStoppingState

        state = EarlyStoppingState(patience=3, min_delta=0.001)
        metrics = [0.7000, 0.7005, 0.7007, 0.7008]

        stopped = False
        for index, metric in enumerate(metrics, start=1):
            state.update(metric, current_iter=index * 1000)
            if state.should_stop:
                stopped = True
                break

        self.assertTrue(stopped)
        self.assertEqual(state.best_iter, 1000)
        self.assertAlmostEqual(state.best_score, 0.7000, places=6)


if __name__ == "__main__":
    unittest.main()
