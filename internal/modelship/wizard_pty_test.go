//go:build linux || darwin

package modelship

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"testing"
	"time"
)

// Python owns the PTY and checks exact termios restoration; the child runs the
// real wizard functions with injected, CPU-only side-effect boundaries. No
// production fixture switches or network/GPU access are needed.
func TestWizardSelectorPTY(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 required for PTY tests")
	}
	cmd := exec.Command(python, filepath.Join("..", "..", "tests", "wizard_selector_pty.py"), os.Args[0])
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("PTY: %v\n%s", err, output)
	} else {
		t.Log(string(output))
	}
}

func TestWizardSelectorHelperProcess(t *testing.T) {
	mode := os.Getenv("MODELSHIP_SELECTOR_TEST_HELPER")
	if mode == "" {
		return
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if mode == "deadline" {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 200*time.Millisecond)
		defer cancel()
	}
	if mode == "wizard" || mode == "wizard-logs" {
		d, calls, _ := testWizardDeps(t, textFixture())
		inspect, detect, cache, keys := 0, 0, 0, 0
		plannedGPU := -1
		originalBuild := d.build
		d.build = func(m Model, h Host, o Options) (Plan, error) { plannedGPU = o.GPU; return originalBuild(m, h, o) }
		original := d.inspect
		d.inspect = func(ctx context.Context, id, rev string) (Model, error) { inspect++; return original(ctx, id, rev) }
		d.detect = func(context.Context, string) (Host, error) {
			detect++
			h := hostFixture()
			g := h.GPUs[0]
			g.Index = 1
			h.GPUs = append(h.GPUs, g)
			return h, nil
		}
		d.cache = func(*Host, Model) { cache++ }
		d.createKey = func(string) error { keys++; return nil }
		if mode == "wizard-logs" {
			d.followLogs = func(ctx context.Context, backend, name string) error {
				if backend != "native" || name != "fixture-name" {
					return fmt.Errorf("wrong fixture log target")
				}
				cmd := exec.CommandContext(ctx, "tail", "-n", "200", "-f", os.Getenv("MODELSHIP_FIXTURE_LOG"))
				cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
				return cmd.Run()
			}
		}
		err := runWizard(ctx, os.Stdin, os.Stdout, d)
		fmt.Printf("FIXTURE_SUMMARY deploys=%d keys=%d inspect=%d detect=%d cache=%d gpu=%d error=%t\n", *calls, keys, inspect, detect, cache, plannedGPU, err != nil)
		return
	}
	s := &wizardSession{terminal: os.Stdin, terminalOutput: os.Stdout, plain: mode == "plain"}
	w := wizardUI{ctx: ctx, session: s, out: os.Stdout}
	m := menuFixture()
	if mode == "no-default" {
		m.defaultValue = ""
	}
	value, err := w.selectChoice("Fixture fixed choices", m.choices, m.defaultValue, false)
	fmt.Printf("FIXTURE_RESULT value=%q history=%q error=%t\n", value, s.answers, err != nil)
}
