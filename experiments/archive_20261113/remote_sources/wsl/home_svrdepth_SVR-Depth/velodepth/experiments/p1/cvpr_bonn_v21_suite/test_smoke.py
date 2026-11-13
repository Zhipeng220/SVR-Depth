import unittest
from smoke import units

class SmokePlanTest(unittest.TestCase):
    def test_smoke_is_one_seed_and_not_formal(self):
        u=units();self.assertEqual(len(u),7);self.assertEqual(len([x for x in u if x['experiment']=='M04']),4)
        self.assertEqual({x['seed'] for x in u if x['family']=='P'},{'20261025'})
        self.assertNotIn('O_fast',{x['family'] for x in u})

if __name__=='__main__':unittest.main()
