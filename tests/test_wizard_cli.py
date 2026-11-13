"""Actual CLI terminal behavior; cancel before any metadata/GPU inspection."""
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import time
import termios
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

    def test_bare_escape_back_and_ctrl_c_restore_terminal_without_network(self):
        master,slave=pty.openpty()
        original=termios.tcgetattr(slave)
        process=subprocess.Popen([BINARY,'launch'],stdin=slave,stdout=slave,stderr=slave)
        data=bytearray()
        def read_until(marker):
            deadline=time.monotonic()+5
            while marker not in data and time.monotonic()<deadline:
                ready,_,_=select.select([master],[],[],0.1)
                if ready:
                    try: chunk=os.read(master,65536)
                    except OSError: break
                    if not chunk: break
                    data.extend(chunk)
            self.assertIn(marker,data)
        try:
            read_until(b'Hugging Face model page')
            os.write(master,b'org/model\r')
            read_until(b'Model revision (branch, tag or commit) [main]')
            os.write(master,b'\x1b')
            read_until(b'Later selections/approvals were discarded')
            read_until(b'Hugging Face model page or org/repo [org/model]')
            self.assertNotIn(b'Inspecting repository',data)
            os.write(master,b'\x03')
            self.assertNotEqual(process.wait(timeout=5),0)
            self.assertEqual(termios.tcgetattr(slave),original)
        finally:
            if process.poll() is None: process.kill();process.wait(timeout=5)
            os.close(master);os.close(slave)


if __name__ == '__main__':
    unittest.main()
