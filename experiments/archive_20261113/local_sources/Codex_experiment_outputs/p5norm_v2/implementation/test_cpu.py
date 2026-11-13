import unittest
from p5norm_v2_policy import select
class TestPolicy(unittest.TestCase):
 def test_selection(self):
  a=[object() for _ in range(4)];b=[object() for _ in range(4)]
  self.assertIs(select('A',a,b),a);self.assertIs(select('C',a,b),b)
 def test_invalid(self):
  with self.assertRaises(ValueError):select('B',[1]*4,[2]*4)
  with self.assertRaises(ValueError):select('A',[1],[2])
if __name__=='__main__':unittest.main()
