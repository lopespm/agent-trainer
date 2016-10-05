from unittest import TestCase

from agent.utils.utils import LinearInterpolator


class TestLinearInterpolator(TestCase):
    def test_interpolate(self):
        interpolator = LinearInterpolator()
        self.assertEqual( 1.30, round(interpolator.interpolate(x=0, x0=1, x1=4, y0=1.0, y1=0.1), 2))
        self.assertEqual( 1.00, round(interpolator.interpolate(x=1, x0=1, x1=4, y0=1.0, y1=0.1), 2))
        self.assertEqual( 0.70, round(interpolator.interpolate(x=2, x0=1, x1=4, y0=1.0, y1=0.1), 2))
        self.assertEqual( 0.40, round(interpolator.interpolate(x=3, x0=1, x1=4, y0=1.0, y1=0.1), 2))
        self.assertEqual( 0.10, round(interpolator.interpolate(x=4, x0=1, x1=4, y0=1.0, y1=0.1), 2))
        self.assertEqual(-0.20, round(interpolator.interpolate(x=5, x0=1, x1=4, y0=1.0, y1=0.1), 2))

    def test_interpolate_with_clip(self):
        interpolator = LinearInterpolator()
        self.assertEqual( 1.00, round(interpolator.interpolate_with_clip(x=0, x0=1, x1=4, y0=1.0, y1=0.1), 2))
        self.assertEqual( 1.00, round(interpolator.interpolate_with_clip(x=1, x0=1, x1=4, y0=1.0, y1=0.1), 2))
        self.assertEqual( 0.70, round(interpolator.interpolate_with_clip(x=2, x0=1, x1=4, y0=1.0, y1=0.1), 2))
        self.assertEqual( 0.40, round(interpolator.interpolate_with_clip(x=3, x0=1, x1=4, y0=1.0, y1=0.1), 2))
        self.assertEqual( 0.10, round(interpolator.interpolate_with_clip(x=4, x0=1, x1=4, y0=1.0, y1=0.1), 2))
        self.assertEqual( 0.10, round(interpolator.interpolate_with_clip(x=5, x0=1, x1=4, y0=1.0, y1=0.1), 2))