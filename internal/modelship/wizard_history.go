package modelship

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
)

var errWizardBack = errors.New("back to previous selection")

type wizardSession struct {
	reader *bufio.Reader

	answers            []string
	cursor             int
	redo               string
	redoSet, committed bool
	terminal           *os.File
}
type replayOutput struct {
	out io.Writer
	s   *wizardSession
}

func (w replayOutput) Write(p []byte) (int, error) {
	if w.s.cursor < len(w.s.answers) {
		return len(p), nil
	}
	return w.out.Write(p)
}
func copyWizardModel(m Model) Model {
	if m.GGUF != nil {
		s := *m.GGUF
		if s.Vision != nil {
			v := *s.Vision
			s.Vision = &v
		}
		m.GGUF = &s
	}
	return m
}
func runWizard(ctx context.Context, input io.Reader, output io.Writer, d wizardDeps) error {
	session := &wizardSession{reader: bufio.NewReaderSize(input, 8193)}
	if f, ok := input.(*os.File); ok && terminalFile(f) {
		if o, ok := output.(*os.File); ok && terminalFile(o) {
			session.terminal = f
		}
	}
	// Only immutable metadata is cached across Back. GPU/cache/fit checks refresh;
	// changing an answer truncates all subsequent answers and approvals.
	inspect := d.inspect
	models := map[string]Model{}
	d.inspect = func(ctx context.Context, id, rev string) (Model, error) {
		key := id + "@" + rev
		if m, ok := models[key]; ok {
			return copyWizardModel(m), nil
		}
		m, e := inspect(ctx, id, rev)
		if e == nil {
			models[key] = copyWizardModel(m)
		}
		return copyWizardModel(m), e
	}
	gguf := d.gguf
	quants := map[string]GGUFSpec{}
	if gguf != nil {
		d.gguf = func(ctx context.Context, m *Model, file string) error {
			key := m.Hub.ID + "@" + m.Hub.SHA + ":" + file
			if s, ok := quants[key]; ok {
				copy := s
				m.GGUF = &copy
				return nil
			}
			if e := gguf(ctx, m, file); e != nil {
				return e
			}
			if m.GGUF != nil {
				quants[key] = *m.GGUF
			}
			return nil
		}
	}
	for {
		session.cursor = 0
		err := runWizardAttempt(ctx, input, replayOutput{output, session}, d, session)
		if !errors.Is(err, errWizardBack) || session.committed {
			return err
		}
		if len(session.answers) == 0 {
			fmt.Fprintln(output, "Already at the first prompt. Ctrl-C cancels setup.")
			continue
		}
		session.redo = session.answers[len(session.answers)-1]
		session.redoSet = true
		session.answers = session.answers[:len(session.answers)-1]
		fmt.Fprintln(output, "Back — editing the previous answer. Later selections/approvals were discarded.")
	}
}
