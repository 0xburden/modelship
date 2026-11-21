//go:build !linux

package modelship

import "os/exec"

// Production admission refuses non-Linux hosts before constructing a command.
func guardDiffusionChild(_ *exec.Cmd) func() { return func() {} }
