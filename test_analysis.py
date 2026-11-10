import unittest
import numpy as np
from analysis import *

class Tests(unittest.TestCase):
    def test_known_shifts(self):
        a=phantom()
        for shift in [(0,0),(7,-11),(-17,13)]:
            s,b=register(a,np.roll(a,shift,(0,1)));self.assertEqual(s,tuple(-v for v in shift));np.testing.assert_allclose(a,b,atol=1e-12)
    def test_noise(self):
        a=phantom();b=np.roll(a,(7,-11),(0,1))+np.random.default_rng(42).normal(0,.002,a.shape);s,c=register(a,b);self.assertEqual(s,(-7,11));self.assertLess(np.sqrt(np.mean((a-c)**2)),.003)
    def test_flat(self):
        with self.assertRaises(ValueError):register(np.ones((16,16)),np.ones((16,16)))
    def test_mismatched(self):
        with self.assertRaises(ValueError):register(phantom(),np.ones((12,12)))
