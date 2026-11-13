import unittest
from smoke import units

class SmokePlanTest(unittest.TestCase):
    def test_smoke_is_four_methods_one_seed(self):
        u=units();self.assertEqual(len(u),4)
        self.assertEqual({x["family"] for x in u},{"A","O_fast","O_multi","P"})
        self.assertTrue(all(x["k"]==16 and x["experiment"]=="M01_M02" for x in u))

if __name__=="__main__":unittest.main()
