//go:build !linux && !darwin

package modelship

import (
	"context"
	"fmt"
	"io"
	"os"
)

func readWizardTerminal(context.Context, *os.File, io.Writer, string) (string, error) {
	return "", fmt.Errorf("interactive terminal input is unsupported on this platform")
}
