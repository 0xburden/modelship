"""CPU-only Go test helper PTYs. Invoked by TestWizardSelectorPTY, not discovery."""
import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import tempfile
import time


def controlling_terminal():
    os.setsid()
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


class Terminal:
    def __init__(self, mode="menu", term="xterm-256color", size=(24, 80), extra_env=None, controlling=False):
        self.master, self.slave = pty.openpty()
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", *size, 0, 0))
        self.original = termios.tcgetattr(self.slave)
        self.data = bytearray()
        self.process = subprocess.Popen(
            [sys.argv[1], "-test.run=^TestWizardSelectorHelperProcess$"],
            stdin=self.slave, stdout=self.slave, stderr=self.slave,
            env={**os.environ, **(extra_env or {}), "MODELSHIP_SELECTOR_TEST_HELPER": mode, "TERM": term},
            preexec_fn=controlling_terminal if controlling else None,
        )

    def expect(self, marker, start=0):
        deadline = time.monotonic() + 8
        while marker not in self.data[start:] and time.monotonic() < deadline:
            ready, _, _ = select.select([self.master], [], [], 0.1)
            if ready:
                self.data.extend(os.read(self.master, 65536))
        assert marker in self.data[start:], (marker, bytes(self.data))

    def send(self, value):
        start = len(self.data)
        os.write(self.master, value)
        return start

    def finish(self, expected):
        self.expect(expected)
        assert self.process.wait(timeout=5) == 0, bytes(self.data)
        assert termios.tcgetattr(self.slave) == self.original, "terminal state leaked"

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        os.close(self.master)
        os.close(self.slave)


def case(keys, expected, mode="menu", **kwargs):
    terminal = Terminal(mode, **kwargs)
    try:
        marker = b"Selection:" if mode != "plain" and kwargs.get("term", "xterm") != "dumb" and kwargs.get("size", (24, 80))[1] >= 50 else b"(number, letter, or full value)"
        terminal.expect(marker)
        if marker == b"Selection:":
            assert b" / m)" not in terminal.data and b"number, letter, or full value" not in terminal.data, bytes(terminal.data)
        terminal.send(keys)
        terminal.finish(expected)
    finally:
        terminal.close()


def main():
    case(b"\x1b[B\r", b'value="choice-14" history=["choice-14"] error=false')
    case(b"\x1bOA\r", b'value="choice-12" history=["choice-12"] error=false')
    case(b"\r", b'value="choice-13"')
    case(b"10\r", b'value="choice-10"')
    case(b"M\r", b'value="choice-13"')
    case(b"CHOICE-25\r", b'value="choice-25"')
    case(b"2\x7f5\r", b'value="choice-05"')
    case(b"10\x1b[B\r", b'value="choice-14"')  # arrows explicitly clear a typed shortcut
    case(b"\x1b[B" * 30 + b"\r", b'value="choice-25"')
    case(b"\x1b[A" * 30 + b"\r", b'value="choice-01"')
    case(b"\x1b[B\r", b'value="choice-01"', mode="no-default")
    case(b"\x1b[Z\r", b'value="choice-13"')  # unknown CSI cannot become Back/approval
    for keys in (b"\x1b", b":back\r", b"\x03", b"\x04", b"\xff\r", b"x" * 8193):
        case(keys, b'value="" history=[] error=true')
    for kwargs in ({"mode": "plain"}, {"term": "dumb"}, {"term": "unknown"}, {"size": (10, 40)}):
        terminal = Terminal(**kwargs)
        try:
            terminal.expect(b"(number, letter, or full value)")
            terminal.send(b"14\r")
            terminal.finish(b'value="choice-14"')
            assert b"\x1b[2K" not in terminal.data, terminal.data
        finally:
            terminal.close()
    for mode in ("deadline", "menu"):
        terminal = Terminal(mode)
        try:
            terminal.expect(b"Selection:")
            if mode == "menu": terminal.process.send_signal(signal.SIGINT)
            terminal.finish(b'value="" history=[] error=true')
        finally:
            terminal.close()
    terminal = Terminal()
    try:
        terminal.expect(b"Selection:")
        terminal.send(b"\x1b")
        # Fragmented CSI within the existing 35 ms continuation window.
        time.sleep(0.005)
        terminal.send(b"[")
        time.sleep(0.005)
        terminal.send(b"B\r")
        terminal.finish(b'value="choice-14"')
    finally:
        terminal.close()
    terminal = Terminal()
    try:
        terminal.expect(b"Selection:")
        fcntl.ioctl(terminal.slave, termios.TIOCSWINSZ, struct.pack("HHHH", 10, 40, 0, 0))
        terminal.send(b"\r")
        terminal.finish(b'value="" history=[] error=true')
    finally:
        terminal.close()
    terminal = Terminal(mode="no-default")
    try:
        terminal.expect(b"Selection:")
        start = terminal.send(b"\r")
        terminal.expect(b"Select an option", start)
        terminal.send(b"25\r")
        terminal.finish(b'value="choice-25"')
    finally:
        terminal.close()
    terminal = Terminal(mode="wizard")
    try:
        terminal.expect(b"Hugging Face model page")
        terminal.send(b"org/model\r")
        terminal.expect(b"Model revision (branch, tag or commit)")
        terminal.send(b"\r")
        terminal.expect(b"Backend\r\n")
        terminal.send(b"\r")
        terminal.expect(b"GPU index\r\n")
        terminal.send(b"\x1b[B\r")
        terminal.expect(b"Deployment name")
        terminal.send(b"\x1b")
        terminal.expect(b"Later selections/approvals were discarded")
        # Back refreshes discovery and restores GPU1 as the highlighted choice.
        terminal.expect(b"> (o) GPU 1 (default)")
        start = terminal.send(b"\x1b[A\r")
        terminal.expect(b"Deployment name", start)
        terminal.send(b"fixture-name\r")
        terminal.expect(b"Inference API port")
        terminal.send(b"\r")
        terminal.expect(b"Context window (tokens)")
        terminal.send(b"\r")
        terminal.expect(b"Existing CUDA Python executable")
        terminal.send(b"\r")
        terminal.expect(b"API access\r\n")
        terminal.send(b"\r")
        terminal.expect(b"API key\r\n")
        terminal.send(b"\r")
        terminal.expect(b"Worker readiness wait")
        terminal.send(b"\r")
        terminal.expect(b"REVIEW")
        terminal.expect(b"(y/N) [n]")
        terminal.send(b"\r")
        terminal.finish(b"FIXTURE_SUMMARY deploys=0 keys=0 inspect=1 detect=2 cache=1 gpu=0 error=true")
    finally:
        terminal.close()
    # Actual cooked-mode Ctrl-C on a controlling PTY stops only the viewer.
    # A separately sessioned CPU fixture keeps writing after the wizard exits.
    for wait_value in (b"0\r", b"\r"):
        with tempfile.TemporaryDirectory() as directory:
            log = os.path.join(directory, "service.log")
            with open(log, "wb") as output:
                worker = subprocess.Popen(
                    [sys.executable, "-u", "-c", "import time\nwhile True:\n print('fixture worker heartbeat', flush=True)\n time.sleep(0.05)"],
                    stdout=output, stderr=output, start_new_session=True,
                )
            terminal = Terminal(mode="wizard-logs", controlling=True, extra_env={"MODELSHIP_FIXTURE_LOG": log})
            try:
                for marker, answer in (
                    (b"Hugging Face model page", b"org/model\r"),
                    (b"Model revision (branch, tag or commit)", b"\r"),
                    (b"Backend\r\n", b"\r"),
                    (b"GPU index\r\n", b"\r"),
                    (b"Deployment name", b"fixture-name\r"),
                    (b"Inference API port", b"\r"),
                    (b"Context window (tokens)", b"\r"),
                    (b"Existing CUDA Python executable", b"\r"),
                    (b"API access\r\n", b"\r"),
                    (b"API key\r\n", b"\r"),
                    (b"Worker readiness wait", wait_value),
                    (b"(y/N) [n]", b"y\r"),
                ):
                    terminal.expect(marker)
                    terminal.send(answer)
                terminal.expect(b"Live logs")
                terminal.expect(b"Ctrl-C to exit")
                terminal.expect(b"fixture worker heartbeat")
                terminal.send(b"\x03")
                terminal.finish(b"FIXTURE_SUMMARY deploys=1 keys=0 inspect=1 detect=1 cache=1 gpu=0 error=false")
                assert b"Exited log viewer. No stop request was sent." in terminal.data, bytes(terminal.data)
                assert worker.poll() is None, "Ctrl-C stopped the retained worker"
            finally:
                terminal.close()
                worker.terminate()
                worker.wait(timeout=5)
    print("Wizard PTYs: selectors, scrolling, shortcuts, fallback, cancellation, replay and terminal restoration passed")


if __name__ == "__main__":
    main()
