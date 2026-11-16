package modelship

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"
	"testing"
)

func menuFixture() *wizardMenu {
	choices := []string{}
	for i := 1; i <= 25; i++ {
		choices = append(choices, fmt.Sprintf("choice-%02d", i))
	}
	return &wizardMenu{label: "Fixed choice", choices: choices, defaultValue: "choice-13", selected: 12, width: 59, visible: 7}
}

func TestWizardMenuViewportAndBoundaries(t *testing.T) {
	m := menuFixture()
	rows := m.rows("")
	if len(rows) != 11 || !strings.Contains(strings.Join(rows, "\n"), "> (o) choice-13 (default)") || rows[1] != "Options 10-16 of 25" {
		t.Fatal(rows)
	}
	m.move(100)
	if m.selected != 24 || m.rows("")[1] != "Options 19-25 of 25" {
		t.Fatal(m)
	}
	m.move(-100)
	if m.selected != 0 {
		t.Fatal(m)
	}
	m.selected = -1
	m.move(1)
	if m.selected != 0 {
		t.Fatal("no-default navigation", m)
	}
	m.selected = -1
	m.move(-1)
	if m.selected != 24 {
		t.Fatal("no-default navigation", m)
	}
}

func TestWizardMenuDisplayBoundedAndCanonicalValuesUntouched(t *testing.T) {
	m := menuFixture()
	m.labels = map[string]string{"choice-13": "hostile\x1b[2J\r\n\u202e\u009b" + strings.Repeat("界", 100)}
	rows := m.rows(strings.Repeat("x", 8192))
	for _, row := range rows {
		if len(row) > m.width || strings.ContainsAny(row, "\x1b\r\n\u202e\u009b") {
			t.Fatal("unsafe/wrapping row", row)
		}
	}
	if m.choices[m.selected] != "choice-13" {
		t.Fatal("display changed value")
	}
	var out bytes.Buffer
	m.draw(&out, "")
	first := out.String()
	out.Reset()
	m.move(1)
	m.draw(&out, "")
	if strings.Contains(first, "\x1b[?25") || !strings.HasPrefix(out.String(), "\r\x1b[10A") {
		t.Fatal("cursor ownership", out.String())
	}
}

func TestWizardMenuTermFallbacks(t *testing.T) {
	for _, value := range []string{"", "dumb", "unknown", "xterm\x1b[2J"} {
		if wizardMenuTERM(value) {
			t.Fatal(value)
		}
	}
	for _, value := range []string{"xterm-256color", "screen", "tmux-256color", "linux", "vt100"} {
		if !wizardMenuTERM(value) {
			t.Fatal(value)
		}
	}
	if newWizardMenu(nil, "value", []string{"one"}, nil, "one") != nil || newWizardMenu(&wizardSession{plain: true}, "value", []string{"one"}, nil, "one") != nil {
		t.Fatal("nonterminal enabled")
	}
}

func TestWizardMenuShortcutsAndCanonicalHistory(t *testing.T) {
	for input, want := range map[string]string{"10": "choice-10", "M": "choice-13", "CHOICE-25": "choice-25", "": "choice-13", "bad\n14": "choice-14"} {
		m := menuFixture()
		s := &wizardSession{}
		var out bytes.Buffer
		w := wizardUI{ctx: context.Background(), session: s, in: bufio.NewReader(strings.NewReader(input + "\n")), out: &out}
		got, err := w.selectChoice("Choice", m.choices, "choice-13", false)
		if err != nil || got != want || len(s.answers) != 1 || s.answers[0] != want {
			t.Fatal(input, got, err, s.answers)
		}
	}
	// Replay validates canonical values without touching input, even if the UI
	// would otherwise be interactive. A removed choice invalidates downstream.
	s := &wizardSession{answers: []string{"choice-13"}}
	w := wizardUI{ctx: context.Background(), session: s, out: io.Discard}
	if got, err := w.selectChoice("Choice", menuFixture().choices, "choice-01", false); err != nil || got != "choice-13" {
		t.Fatal(got, err)
	}
	s = &wizardSession{answers: []string{"removed", "old-approval"}}
	w = wizardUI{ctx: context.Background(), session: s, out: io.Discard, in: bufio.NewReader(strings.NewReader("2\n"))}
	if got, err := w.selectChoice("Choice", []string{"new-one", "new-two"}, "new-one", false); err != nil || got != "new-two" || len(s.answers) != 1 {
		t.Fatal(got, err, s.answers)
	}
}

func TestWizardBackNeverRestoresPreviousApproval(t *testing.T) {
	for _, old := range []string{"y", "yes", "n"} {
		s := &wizardSession{redo: old, redoSet: true}
		var out bytes.Buffer
		w := wizardUI{ctx: context.Background(), session: s, in: bufio.NewReader(strings.NewReader("\n")), out: &out}
		approved, err := w.confirm("Explicit safety consent")
		if err != nil || approved || !strings.Contains(out.String(), "[n]") || s.answers[0] != "n" {
			t.Fatal(old, approved, err, out.String())
		}
	}
}

func TestWizardMenuCleanRadioDisplayAndPlainShortcuts(t *testing.T) {
	m := &wizardMenu{label: "Input mode", choices: []string{"text-only", "vision-images"}, defaultValue: "text-only", selected: 0, width: 79, visible: 7}
	rows := m.rows("")
	display := strings.Join(rows, "\n")
	if rows[0] != "Input mode" || !strings.Contains(display, "> (o) text-only (default)") || !strings.Contains(display, "  ( ) vision-images") {
		t.Fatal(display)
	}
	for _, unwanted := range []string{"1 / a)", "2 / b)", "number", "letter", "Options", "Shortcut"} {
		if strings.Contains(display, unwanted) {
			t.Fatal("radio clutter", unwanted, display)
		}
	}
	var out bytes.Buffer
	w := wizardUI{ctx: context.Background(), session: &wizardSession{plain: true}, in: bufio.NewReader(strings.NewReader("b\n")), out: &out}
	got, err := w.selectChoice("Input mode", m.choices, "text-only", false)
	if err != nil || got != "vision-images" || !strings.Contains(out.String(), "1 / a) text-only") || !strings.Contains(out.String(), "(number, letter, or full value)") {
		t.Fatal(got, err, out.String())
	}
}
