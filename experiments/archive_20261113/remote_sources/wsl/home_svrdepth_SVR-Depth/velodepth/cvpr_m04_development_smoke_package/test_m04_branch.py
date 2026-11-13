import unittest
from copy import deepcopy
from m04_branch import branch_step

class BranchTest(unittest.TestCase):
    def test_each_arm_and_refresh(self):
        native=('r0','c0',['f0'],'rays'); repaired=('r1','c1',['f1'],'rays')
        def make(calls):
            def step(repeats):
                calls.append(repeats)
                return (native,'D0') if repeats==0 else (repaired,'D1')
            return step
        expected={'OFF':(native,'D0',[0]),'OUTPUT_ONLY':(native,'D1',[0,1]),
                  'F_ONLY':(('r0','c0',['f1'],'rays'),'D0',[0,1]),'P_NATIVE':(repaired,'D1',[1])}
        for arm,(state,depth,expected_calls) in expected.items():
            calls=[]; s,d,n=branch_step(arm,1,16,make(calls),deepcopy)
            self.assertEqual((s,d,calls),(state,depth,expected_calls)); self.assertEqual(n,len(calls))
            calls=[]; s,d,n=branch_step(arm,16,16,make(calls),deepcopy)
            self.assertEqual((s,d,n,calls),(native,'D0',0,[0]))
        with self.assertRaises(ValueError): branch_step('wrong',1,16,make([]),deepcopy)

if __name__=='__main__': unittest.main()
