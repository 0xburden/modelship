package modelship

import (
	"os/exec"
	"runtime"
	"syscall"
)

func guardDiffusionChild(cmd *exec.Cmd) func() {
	// Linux ties the death signal to the creating kernel thread, not just
	// the process. Keep that thread alive until the child is reaped.
	runtime.LockOSThread()
	cmd.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
	return runtime.UnlockOSThread
}
