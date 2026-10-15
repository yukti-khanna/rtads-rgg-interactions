"""Independent campaign regression tests, including blocked-draw and 100-system paths."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'reviewer_analysis/control_simulation_manifest'
EXCLUSION=BASE/'simulation_pair_names_400.txt'

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

class CampaignTests(unittest.TestCase):
    def validate_files(self, directory, campaign, count, expected_sequences):
        pairs=(directory/f'{campaign}_pairs_{count}_new.txt').read_text().splitlines()
        manifest=pd.read_csv(directory/f'{campaign}_manifest_{count}_new.csv')
        records={}
        current=None
        for line in (directory/f'{campaign}_sequences_{count}_new.fasta').read_text().splitlines():
            if line.startswith('>'):
                current=line[1:]
                self.assertNotIn(current,records)
                records[current]=''
            else: records[current]+=line
        self.assertEqual(len(pairs),count)
        self.assertEqual(len(set(pairs)),count)
        self.assertEqual(pairs,manifest.name.tolist())
        self.assertEqual(manifest.columns[0],'name')
        self.assertFalse(set(pairs)&set(EXCLUSION.read_text().splitlines()))
        components=set()
        for name in pairs:
            self.assertEqual(name.count('__'),1)
            r,t=name.split('__')
            self.assertTrue(r.startswith('RGG_') and t.startswith('TAD_'))
            components.update((r,t))
        self.assertEqual(set(records),components)
        self.assertEqual(len(records),expected_sequences)
        self.assertFalse(manifest.already_run_or_running.any())
        for row in manifest.itertuples():
            self.assertEqual(row.TAD_sequence,records[row.TAD_variant])
            self.assertEqual(row.RGG_sequence,records[row.RGG_variant])
        return manifest

    def test_final_campaigns(self):
        self.assertEqual(digest(EXCLUSION),json.loads((BASE/'control_selection_audit_final.json').read_text())['exclusion_sha256'])
        m=self.validate_files(BASE,'shuffle',400,90)
        self.assertEqual(m.condition.value_counts().to_dict(),{'TAD_SHUFFLE':320,'RGG_SHUFFLE':80})
        t=m[m.condition=='TAD_SHUFFLE']; r=m[m.condition=='RGG_SHUFFLE']
        self.assertEqual(t.TAD_variant.nunique(),64)
        self.assertTrue(t.groupby('TAD_variant').RGG_parent.nunique().eq(5).all())
        self.assertTrue(t.groupby('TAD_parent').TAD_variant.nunique().eq(4).all())
        self.assertEqual(r.RGG_variant.nunique(),5)
        self.assertTrue(r.groupby('RGG_variant').TAD_parent.nunique().eq(16).all())
        self.assertEqual(m.RGG_UniProt.nunique(),5)
        self.assertNotIn('basic_AD_control',set(m.TAD_class))
        b=self.validate_files(BASE,'basic_control',200,54)
        self.assertEqual(set(b.TAD_gene),{'TCF7L2','TET3','EOMES','LEF1'})
        self.assertTrue(b.groupby('TAD_parent').size().eq(50).all())
        self.assertEqual(b.RGG_UniProt.nunique(),50)
        self.assertEqual(len(pd.read_csv(BASE/'shuffle_WT_reference_status.csv')),80)

    def test_reproducibility_and_100_option(self):
        with tempfile.TemporaryDirectory(prefix='campaign-tests-') as tmp:
            out=Path(tmp)
            for script,campaign,count,seqs in [('03_select_new_shuffle_400.py','shuffle',400,90),
                                              ('04_select_new_basic_controls.py','basic_control',200,54)]:
                subprocess.run([sys.executable,str(ROOT/script),'--output-dir',str(out)],cwd=ROOT,check=True,capture_output=True)
                self.validate_files(out,campaign,count,seqs)
                self.assertEqual(digest(out/f'{campaign}_pairs_{count}_new.txt'),digest(BASE/f'{campaign}_pairs_{count}_new.txt'))
                self.assertEqual(digest(out/f'{campaign}_sequences_{count}_new.fasta'),digest(BASE/f'{campaign}_sequences_{count}_new.fasta'))
            subprocess.run([sys.executable,str(ROOT/'04_select_new_basic_controls.py'),'--n-systems','100','--output-dir',str(out)],cwd=ROOT,check=True,capture_output=True)
            m=self.validate_files(out,'basic_control',100,29)
            self.assertTrue(m.groupby('TAD_parent').size().eq(25).all())

    def test_rejection_uses_run_status_and_aborts(self):
        with tempfile.TemporaryDirectory(prefix='campaign-conflict-') as tmp:
            work=Path(tmp)
            for name in ['native_RTAD_eligible_pool.csv','external_acidic_AD_parents.csv','eligible_conventional_RGGs.csv']:
                shutil.copyfile(BASE/name,work/name)
            df=pd.read_parquet(BASE/'shuffle_candidate_universe.parquet')
            blocked=set(df[df.condition=='RGG_SHUFFLE'].pair_name)
            ex=work/'existing.txt'
            ex.write_text('\n'.join(sorted(blocked))+'\n')
            df['already_run_or_running']=df.pair_name.isin(blocked)
            df.to_parquet(work/'shuffle_candidate_universe.parquet',index=False)
            (work/'candidate_exclusion_metadata.json').write_text(json.dumps({'exclusion_sha256':digest(ex),'number_loaded':len(blocked)}))
            result=subprocess.run([sys.executable,str(ROOT/'03_select_new_shuffle_400.py'),'--input-dir',str(work),
                '--output-dir',str(work),'--existing-pairs',str(ex),'--max-draws','2'],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('no feasible common RGG set',result.stderr)
            audit=json.loads((work/'shuffle_rejected_draws.json').read_text())
            self.assertEqual(len(audit),2)
            self.assertFalse((work/'shuffle_pairs_400_new.txt').exists())

    def test_full_pool_and_existing_classification(self):
        native=pd.read_csv(BASE/'native_RTAD_eligible_pool.csv')
        self.assertNotIn('selected',native.columns)
        variants=pd.read_csv(BASE/'shuffle_TAD_variants.csv')
        self.assertEqual(len(variants),(len(native)+3)*20)
        self.assertNotIn('basic_AD_control',set(variants['class']))
        self.assertTrue(variants.groupby('parent_id').size().eq(20).all())
        m=pd.read_csv(BASE/'shuffle_manifest_400_new.csv')
        self.assertEqual(m[['TAD_parent','TAD_class']].drop_duplicates().TAD_class.value_counts().to_dict(),
                         {'native_RTAD':13,'external_AD_not_RTAD':3})
        self.assertFalse(m.already_used.any())
        self.assertIn('TAD_shuffle_seed',m.columns)
        self.assertIn('RGG_shuffle_seed',m.columns)
        audit=pd.read_csv(BASE/'existing_400_pair_classification.csv')
        self.assertEqual(len(audit),400)
        self.assertEqual(audit.pair_name.nunique(),400)
        self.assertEqual(audit.category.value_counts().to_dict(),{'EXISTING_SHUFFLE':240,'OTHER_EXISTING':80,
            'EXISTING_UNWANTED_BASIC_SHUFFLE':60,'EXISTING_BASIC_WT':20})
        b=pd.read_csv(BASE/'basic_control_manifest_200_new.csv')
        self.assertFalse(set(m.name)&set(b.name))

if __name__=='__main__': unittest.main()
