//go:build linux

package modelship

import (
	"os"
	"syscall"
	"unsafe"
)

// A character device is not necessarily a terminal (e.g. /dev/null).
func terminalFile(f *os.File) bool {
	var state syscall.Termios
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, f.Fd(), syscall.TCGETS, uintptr(unsafe.Pointer(&state)))
	return errno == 0
}
