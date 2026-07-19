import math
import unittest

from ace_schedule import (
    progress_to_step,
    quadratic_bump,
    quadratic_bump_derivative,
    scheduled_steps,
)


class ACEScheduleTest(unittest.TestCase):
    def test_bump_preserves_endpoints(self):
        self.assertEqual(quadratic_bump(0.0, 5.0), 0.0)
        self.assertEqual(quadratic_bump(1.0, 5.0), 0.0)
        self.assertEqual(quadratic_bump(0.5, 5.0), 1.25)

    def test_derivative_is_continuous_time_derivative(self):
        strength = 5.0
        progress = 0.3
        eps = 1e-6
        finite_difference = (
            quadratic_bump(progress + eps, strength) - quadratic_bump(progress - eps, strength)
        ) / (2.0 * eps)
        self.assertTrue(math.isclose(finite_difference, quadratic_bump_derivative(progress, strength), rel_tol=1e-8))

    def test_paper_resampling_point(self):
        self.assertEqual(progress_to_step(0.3, 50), 15)
        self.assertEqual(progress_to_step(0.5, 50), 25)
        self.assertEqual(scheduled_steps([0.3, 0.3], 50), (15,))

    def test_invalid_progress_is_rejected(self):
        with self.assertRaises(ValueError):
            quadratic_bump(-0.1, 5.0)


if __name__ == "__main__":
    unittest.main()
