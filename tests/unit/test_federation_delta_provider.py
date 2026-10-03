import pathlib,subprocess,tempfile,unittest
from glx.federation.delta_provider import DeltaProviderError,observe_git_range
class FederationDeltaProviderTests(unittest.TestCase):
 def make_repo(self):
  td=tempfile.TemporaryDirectory();p=pathlib.Path(td.name)
  subprocess.run(["git","init","-q"],cwd=p,check=True)
  subprocess.run(["git","config","user.email","fixture@example.invalid"],cwd=p,check=True)
  subprocess.run(["git","config","user.name","Fixture"],cwd=p,check=True)
  (p/"a.py").write_text("def f(x):\\n    return x\\n")
  subprocess.run(["git","add","a.py"],cwd=p,check=True);subprocess.run(["git","commit","-qm","base"],cwd=p,check=True)
  base=subprocess.check_output(["git","rev-parse","HEAD"],cwd=p,text=True).strip()
  (p/"a.py").write_text("import math\\ndef f(x, y=0):\\n    return x+y\\n\\ndef g():\\n    return math.pi\\n")
  subprocess.run(["git","add","a.py"],cwd=p,check=True);subprocess.run(["git","commit","-qm","head"],cwd=p,check=True)
  head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=p,text=True).strip()
  return td,p,base,head
 def test_real_git_range_is_observed_without_glx_side_effects(self):
  td,p,b,h=self.make_repo()
  try:
   v=observe_git_range(p,b,h,repository_ref="fixture/repo",provider_source_ref="glx:federation/v1",process_semantics_ref="chunk-chunk:hmk9d",process_semantics_digest="1"*64)
   self.assertEqual(v.changed_files,("a.py",));self.assertEqual((v.authority_effect,v.mutation_effect),("NONE","NONE"))
   self.assertTrue(dict(v.delta_histogram));self.assertFalse((p/"glitchlab/.glx/delta_report.json").exists())
  finally:td.cleanup()
 def test_exact_head_substitution_changes_digest(self):
  td,p,b,h=self.make_repo()
  try:
   v=observe_git_range(p,b,h,repository_ref="fixture/repo",provider_source_ref="glx:federation/v1",process_semantics_ref="chunk-chunk:hmk9d",process_semantics_digest="1"*64)
   self.assertEqual(v.base_commit,b);self.assertEqual(v.head_commit,h)
   with self.assertRaises(DeltaProviderError):observe_git_range(p,h,h,repository_ref="fixture/repo",provider_source_ref="x",process_semantics_ref="chunk-chunk:hmk9d",process_semantics_digest="1"*64)
  finally:td.cleanup()
 def test_missing_revision_fails_closed(self):
  td,p,b,h=self.make_repo()
  try:
   with self.assertRaises(DeltaProviderError):observe_git_range(p,"0"*40,h,repository_ref="fixture/repo",provider_source_ref="x",process_semantics_ref="chunk-chunk:hmk9d",process_semantics_digest="1"*64)
  finally:td.cleanup()
if __name__=="__main__":unittest.main()
