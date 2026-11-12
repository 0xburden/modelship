"""Release installer URL embedding and checksum tests; fake compiler, no network."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleasePackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'scripts').mkdir()
        (self.root/'tools').mkdir()
        shutil.copyfile(ROOT/'scripts/release.sh',self.root/'scripts/release.sh')
        shutil.copyfile(ROOT/'install.sh',self.root/'install.sh')
        compiler=self.root/'tools/go'
        compiler.write_text('''#!/bin/sh
while [ "$#" -gt 0 ]; do
 case "$1" in -o) shift; out=$1;; esac
 shift
done
printf '%s' fake-compiled-binary > "$out"
''')
        compiler.chmod(0o755)
        self.env={**os.environ,'PATH':str(self.root/'tools')+os.pathsep+os.environ['PATH'],'VERSION':'v9.8.7-test.1','MODELSHIP_PUBLISH_ORIGIN':'https://release.example:8443'}

    def run_build(self):
        return subprocess.run(['sh','scripts/release.sh'],cwd=self.root,env=self.env,text=True,capture_output=True,timeout=10)

    def test_bakes_immutable_base_and_checksums_actual_served_script(self):
        result=self.run_build()
        self.assertEqual(result.returncode,0,result.stderr)
        script=(self.root/'dist/install.sh').read_text()
        self.assertIn("release_default='https://release.example:8443/v9.8.7-test.1'",script)
        self.assertNotIn('__MODELSHIP_RELEASE_BASE__',script)
        for line in (self.root/'dist/SHA256SUMS').read_text().splitlines():
            digest,name=line.split()
            self.assertEqual(digest,hashlib.sha256((self.root/'dist'/name).read_bytes()).hexdigest())
        self.assertIn('__MODELSHIP_RELEASE_BASE__',(self.root/'install.sh').read_text())

    def test_rejects_unsafe_version_or_publish_origin(self):
        for key,value in [('VERSION','v1/../escape'),('VERSION','v1&shell'),('MODELSHIP_PUBLISH_ORIGIN','http://plain.example'),('MODELSHIP_PUBLISH_ORIGIN','https://u:p@release.example'),('MODELSHIP_PUBLISH_ORIGIN','https://release.example/path?x=1')]:
            before=self.env[key]
            self.env[key]=value
            self.assertNotEqual(self.run_build().returncode,0,(key,value))
            self.env[key]=before
        self.assertFalse((self.root/'dist').exists())


if __name__ == '__main__':
    unittest.main()
