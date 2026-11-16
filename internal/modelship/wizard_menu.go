package modelship

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"golang.org/x/term"
)

var errWizardTerminalSize = errors.New("terminal resized too small; restart setup with modelship launch --plain; no deployment or key was created")

// A small, bounded viewport, not a full-screen TUI. No colors, hidden cursor or
// alternate screen: ordinary SSH terminals need no font or mouse support.
type wizardMenu struct {
	label                 string
	defaultValue          string
	choices               []string
	labels                map[string]string
	selected              int
	width, visible, drawn int
}

func wizardMenuTERM(value string) bool {
	for _, prefix := range []string{"xterm", "screen", "tmux", "rxvt", "vt100", "vt220", "ansi", "linux"} {
		if value == prefix || strings.HasPrefix(value, prefix+"-") {
			return true
		}
	}
	return false
}

func newWizardMenu(s *wizardSession, label string, choices []string, labels map[string]string, fallback string) *wizardMenu {
	if s == nil || s.plain || s.terminal == nil || s.terminalOutput == nil || len(choices) == 0 || !wizardMenuTERM(os.Getenv("TERM")) {
		return nil
	}
	width, height, err := term.GetSize(int(s.terminalOutput.Fd()))
	if err != nil || width < 50 || height < 10 {
		return nil
	}
	m := &wizardMenu{label: label, defaultValue: fallback, choices: choices, labels: labels, selected: -1, width: width - 1, visible: min(7, height-5)}
	if value, ok := choiceValue(fallback, choices); ok {
		m.defaultValue = value
		for i, choice := range choices {
			if choice == value {
				m.selected = i
				break
			}
		}
	}
	return m
}

func (m *wizardMenu) move(delta int) {
	if m.selected < 0 {
		if delta < 0 {
			m.selected = len(m.choices) - 1
		} else {
			m.selected = 0
		}
	} else {
		m.selected = max(0, min(len(m.choices)-1, m.selected+delta))
	}
}

// ASCII-only display makes width accounting conservative even for wide glyphs,
// combining characters and hostile terminal controls. Values themselves are
// never truncated, escaped or replaced in the plan/history.
func (m *wizardMenu) row(value string) string {
	value = strings.ReplaceAll(value, "—", "-")
	quoted := strconv.QuoteToASCII(value)
	value = quoted[1 : len(quoted)-1]
	if len(value) > m.width {
		value = value[:m.width-3] + "..."
	}
	return value
}

func (m *wizardMenu) rows(typed string) []string {
	n := min(m.visible, len(m.choices))
	start := max(0, min(m.selected-n/2, len(m.choices)-n))
	rows := []string{m.row(m.label)}
	if len(m.choices) > n {
		rows = append(rows, m.row(fmt.Sprintf("Options %d-%d of %d", start+1, start+n, len(m.choices))))
	}
	for i := start; i < start+n; i++ {
		marker := "  ( )"
		if i == m.selected {
			marker = "> (o)"
		}
		label := m.choices[i]
		if m.labels[label] != "" {
			label = m.labels[label] + " [" + label + "]"
		}
		if m.choices[i] == m.defaultValue {
			label += " (default)"
		}
		rows = append(rows, m.row(fmt.Sprintf("%s %s", marker, label)))
	}
	rows = append(rows, m.row("Up/Down: select  Enter: confirm  Esc: back"), m.row("Selection: "+typed))
	return rows
}

func (m *wizardMenu) draw(out io.Writer, typed string) {
	rows := m.rows(typed)
	if m.drawn > 0 {
		fmt.Fprintf(out, "\r\x1b[%dA", m.drawn-1)
	}
	for i, row := range rows {
		fmt.Fprint(out, "\r\x1b[2K", row)
		if i < len(rows)-1 {
			fmt.Fprint(out, "\r\n")
		}
	}
	m.drawn = len(rows)
}
