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
	return readWizardInput(ctx, f, out, question, nil)
}

func readWizardInput(ctx context.Context, f *os.File, out io.Writer, question string, menu *wizardMenu) (string, error) {
	fd := int(f.Fd())
	old, e := term.MakeRaw(fd)
	if e != nil {
		return "", fmt.Errorf("cannot enable interactive terminal input")
	}
	defer func() {
		fmt.Fprint(out, "\r\n")
		term.Restore(fd, old)
	}()
	if menu == nil {
		fmt.Fprint(out, question)
	} else {
		menu.draw(out, "")
	}
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
		if menu != nil {
			width, height, err := term.GetSize(fd)
			if err != nil || width <= menu.width || height < menu.visible+5 {
				return "", errWizardTerminalSize
			}
		}
		switch b {
		case 3:
			fmt.Fprint(out, "^C")
			return "", errWizardCanceled
		case 4:
			return "", io.EOF
		case '\r', '\n':
			if len(pending) > 0 {
				return "", fmt.Errorf("invalid Unicode input")
			}
			if menu != nil && len(data) == 0 && menu.selected >= 0 {
				return menu.choices[menu.selected], nil
			}
			return string(data), nil
		case 27:
			// A bare Esc edits history. CSI/SS3 sequences are bounded and never
			// interpreted as approval; unknown navigation sequences are ignored.
			sequence, bare, err := readWizardEscape(f, ready)
			if err != nil {
				return "", err
			}
			if bare {
				return "", errWizardBack
			}
			if menu != nil {
				delta := 0
				switch sequence {
				case "[A", "OA":
					delta = -1
				case "[B", "OB":
					delta = 1
				}
				if delta != 0 {
					menu.move(delta)
					data, pending = nil, nil
					menu.draw(out, "")
				}
			}
			continue
		case 8, 127:
			if len(pending) > 0 {
				pending = nil
				continue
			}
			if len(data) > 0 {
				_, size := utf8.DecodeLastRune(data)
				data = data[:len(data)-size]
				if menu == nil {
					fmt.Fprint(out, "\b \b")
				} else {
					menu.draw(out, string(data))
				}
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
		if menu == nil {
			fmt.Fprint(out, string(pending))
		} else {
			menu.draw(out, string(data))
		}
		pending = nil
	}
}

// Distinguish a bare Esc from arrow keys, including application-cursor (SS3)
// mode. Each continuation has a 35 ms deadline and sequences are capped at 24
// bytes. Cancellation/EOF remains authoritative even inside an escape sequence.
func readWizardEscape(f *os.File, ready func(int) (bool, error)) (string, bool, error) {
	sequence := []byte{}
	one := make([]byte, 1)
	for i := 0; i < 24; i++ {
		more, err := ready(35)
		if err != nil {
			return "", false, err
		}
		if !more {
			return string(sequence), len(sequence) == 0, nil
		}
		if _, err := f.Read(one); err != nil {
			return "", false, err
		}
		b := one[0]
		if b == 3 {
			return "", false, errWizardCanceled
		}
		if b == 4 {
			return "", false, io.EOF
		}
		sequence = append(sequence, b)
		if i == 0 {
			if b != '[' && b != 'O' {
				return "", true, nil
			}
		} else if b >= 0x40 && b <= 0x7e {
			return string(sequence), false, nil
		}
	}
	return string(sequence), false, fmt.Errorf("terminal escape sequence exceeded limit")
}
