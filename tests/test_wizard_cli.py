"""Actual CLI terminal behavior; cancel before any metadata/GPU inspection."""
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import time
import unittest

BINARY=os.environ.get('MODELSHIP_TEST_BINARY','')


@unittest.skipUnless(BINARY and sys.platform.startswith('linux'),'set MODELSHIP_TEST_BINARY on Linux for terminal CLI integration')
class WizardCLITests(unittest.TestCase):
    def test_noninteractive_launch_fails_fast_without_hardware_or_network(self):
        result=subprocess.run([BINARY,'launch'],input='',text=True,capture_output=True,timeout=5)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('requires terminal',result.stderr)
        default=subprocess.run([BINARY],input='',text=True,capture_output=True,timeout=5)
        self.assertEqual(default.returncode,0,default.stderr)
        self.assertIn('Commands:',default.stdout)
        self.assertNotIn('Inspecting repository',default.stdout)

    def test_default_and_explicit_tty_launch_cancel_on_sigint(self):
        for args in [[],['launch']]:
            with self.subTest(args=args):
                master,slave=pty.openpty()
                process=subprocess.Popen([BINARY,*args],stdin=slave,stdout=slave,stderr=slave,env={**os.environ,'HF_TOKEN':'fixture-secret-never-displayed'})
                os.close(slave)
                data=bytearray()
                try:
                    deadline=time.monotonic()+10
                    while b'Hugging Face model page' not in data and time.monotonic()<deadline:
                        ready,_,_=select.select([master],[],[],0.2)
                        if ready:
                            try:chunk=os.read(master,65536)
                            except OSError:break
                            if not chunk:break
                            data.extend(chunk)
                    self.assertIn(b'Hugging Face model page',data)
                    self.assertNotIn(b'fixture-secret-never-displayed',data)
                    process.send_signal(signal.SIGINT)
                    self.assertNotEqual(process.wait(timeout=5),0)
                finally:
                    if process.poll() is None:
                        process.kill();process.wait(timeout=5)
                    os.close(master)


if __name__ == '__main__':
    unittest.main()
