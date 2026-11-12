//go:build !linux && !darwin

package modelship

import "os"

func terminalFile(*os.File) bool { return false }
