//go:build linux || darwin

package modelship

import (
	"context"
	"fmt"
	"io"
	"os"
	"unicode"
	"unicode/utf8"

	"golang.org/x/sys/unix"
	"golang.org/x/term"
)

// Raw mode is scoped to one prompt and always restored before metadata work,
// approvals or deployment. Esc edits history; Ctrl-C/EOF cancel, never approve.
func readWizardTerminal(ctx context.Context, f *os.File, out io.Writer, question string) (string, error) {
	fd := int(f.Fd())
	old, e := term.MakeRaw(fd)
	if e != nil {
		return "", fmt.Errorf("cannot enable interactive terminal input")
	}
	defer term.Restore(fd, old)
	fmt.Fprint(out, question)
	ready := func(ms int) (bool, error) {
		p := []unix.PollFd{{Fd: int32(fd), Events: unix.POLLIN}}
		n, e := unix.Poll(p, ms)
		if e == unix.EINTR {
			return false, nil
		}
		if e != nil {
			return false, e
		}
		if n > 0 && p[0].Revents&(unix.POLLHUP|unix.POLLERR|unix.POLLNVAL) != 0 {
			return false, io.EOF
		}
		return n > 0, nil
	}
	data, pending := []byte{}, []byte{}
	one := make([]byte, 1)
	for {
		if e := ctx.Err(); e != nil {
			return "", e
		}
		ok, e := ready(100)
		if e != nil {
			return "", e
		}
		if !ok {
			continue
		}
		if _, e = f.Read(one); e != nil {
			return "", e
		}
		b := one[0]
		switch b {
		case 3:
			fmt.Fprint(out, "^C\r\n")
			return "", errWizardCanceled
		case 4:
			return "", io.EOF
		case '\r', '\n':
			if len(pending) > 0 {
				return "", fmt.Errorf("invalid Unicode input")
			}
			fmt.Fprint(out, "\r\n")
			return string(data), nil
		case 27:
			// Ignore terminal navigation sequences rather than treating arrow keys as
			// Back. A bare Esc is detected immediately, without requiring Enter.
			if more, _ := ready(35); more {
				if _, e = f.Read(one); e != nil {
					return "", e
				}
				if one[0] == 3 {
					fmt.Fprint(out, "^C\r\n")
					return "", errWizardCanceled
				}
				if one[0] == 4 {
					return "", io.EOF
				}
				if one[0] == '[' || one[0] == 'O' {
					for i := 0; i < 24; i++ {
						if more, _ = ready(35); !more {
							break
						}
						if _, e = f.Read(one); e != nil {
							return "", e
						}
						if one[0] == 3 {
							fmt.Fprint(out, "^C\r\n")
							return "", errWizardCanceled
						}
						if one[0] == 4 {
							return "", io.EOF
						}
						if one[0] >= 0x40 && one[0] <= 0x7e {
							break
						}
					}
					continue
				}
			}
			fmt.Fprint(out, "\r\n")
			return "", errWizardBack
		case 8, 127:
			if len(pending) > 0 {
				pending = nil
				continue
			}
			if len(data) > 0 {
				_, size := utf8.DecodeLastRune(data)
				data = data[:len(data)-size]
				fmt.Fprint(out, "\b \b")
			}
			continue
		}
		if b < 32 {
			return "", fmt.Errorf("terminal controls are not accepted")
		}
		pending = append(pending, b)
		if !utf8.FullRune(pending) {
			continue
		}
		r, size := utf8.DecodeRune(pending)
		if r == utf8.RuneError && size == 1 || unicode.IsControl(r) || unicode.Is(unicode.Cf, r) {
			return "", fmt.Errorf("invalid Unicode or terminal controls are not accepted")
		}
		if len(data)+len(pending) > 8192 {
			return "", fmt.Errorf("terminal input exceeded 8 KiB")
		}
		data = append(data, pending...)
		fmt.Fprint(out, string(pending))
		pending = nil
	}
}
