//go:build darwin

package modelship

import (
	"os"
	"syscall"
	"unsafe"
)

func terminalFile(f *os.File) bool {
	var state syscall.Termios
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, f.Fd(), syscall.TIOCGETA, uintptr(unsafe.Pointer(&state)))
	return errno == 0
}
