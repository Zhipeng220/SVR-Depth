import csv
import json
import tempfile
import unittest
from pathlib import Path
from m05_formal_reaggregate import aggregate, indexed, read_csv, reaggregate

class RegressionTests(unittest.TestCase):
    def test_csv_bom_and_plain(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.csv'
            for encoding in ('utf-8','utf-8-sig'):
                p.write_text('seed,value\n1,2\n',encoding=encoding)
                self.assertEqual(read_csv(p,('seed',))[0]['seed'],'1')

    def test_duplicate_key(self):
        with self.assertRaises(ValueError): indexed([{'a':'1'},{'a':'1'}],('a',))

    def test_equal_weights_and_parent_separation(self):
        rows=[]
        for parent,shift in [('A_native',0),('P_maintained',10)]:
            for seq, vals in [('x',[(1,0),(1,0),(4,6)]),('y',[(1,2),(4,4)])]:
                for age,value in vals: rows.append({'seed':'1','sequence':seq,'parent':parent,'age':age,'effect':value+shift})
        result,_=aggregate(rows,['x','y'],[1,4])
        self.assertEqual(result[0]['mean'],3)
        self.assertEqual(result[1]['mean'],13)
        with self.assertRaises(ValueError): aggregate(rows,['x','y','empty'],[1,4])

    def fixture(self, root, reverse=False, missing_gt=False, duplicate=False, value=0.2):
        inp=root/'in'; inp.mkdir()
        (inp/'protocol_lock.json').write_text(json.dumps({'sequences':{'x':'p'},'ages':[1],'selection':{'x':{'selected_cycles':[0]}}}))
        (inp/'m05_formal_summary.json').write_text('{"all_invariants_pass":true}')
        def write(name,rows):
            with (inp/name).open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        anchor=[]; future=[]; current=[]
        for parent in ('A_native','P_maintained'):
            base={'seed':'1','sequence':'x','cycle':'0','age':'1','parent':parent}
            anchor.append(dict(base,H4_T_minus_O_AbsRel=str(value-0.3)))
            for branch in ('O','T'):
                predicted=({'O':value,'T':0.3} if reverse else {'O':0.3,'T':value})[branch]
                future.append(dict(base,branch=branch,horizon='4',frame_index='5',anchor_index='1',gt_matched=str(not missing_gt),valid_pixels='5',total_pixels='10',gt_valid_ratio='0.5',AbsRel=str(predicted)))
                current.append(dict(base,branch=branch,anchor_index='1',current_output_source='D1',AbsRel='0.2',RMSE='0.4',delta1='0.8',valid_pixels='5',total_pixels='10',gt_matched='True'))
        if duplicate: future.append(future[0])
        write('m05_anchor_effects.csv',anchor);write('m05_future_frame_metrics.csv',future);write('m05_current_frame_metrics.csv',current)
        write('anchor_manifest.csv',[{'sequence':'x','cycle':'0','age':'1','future_horizon':'4','future_index':'5','gt_matched':'True'}])
        return inp

    def test_sign_zero_effect_and_no_overwrite(self):
        for value in (0.2,0.3,0.4):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d); inp=self.fixture(root,value=value)
                with self.assertRaisesRegex(ValueError,'BLOCKED'): reaggregate(inp,root/'strict')
                result,_=reaggregate(inp,root/'out',descriptive_only=True)
                self.assertAlmostEqual(result[0]['mean'],value-0.3)
                with self.assertRaises(FileExistsError): reaggregate(inp,root/'out')
                with self.assertRaises(ValueError): reaggregate(inp,inp/'sub')

    def test_reversed_branch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=self.fixture(root,reverse=True)
            with self.assertRaisesRegex(ValueError,'anchor effect mismatch'): reaggregate(inp,root/'out')

    def test_missing_gt_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=self.fixture(root,missing_gt=True)
            with self.assertRaisesRegex(ValueError,'missing GT'): reaggregate(inp,root/'out')

    def test_duplicate_future_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=self.fixture(root,duplicate=True)
            with self.assertRaisesRegex(ValueError,'duplicate'): reaggregate(inp,root/'out')

if __name__=='__main__': unittest.main()
