package modelship

import (
	"bytes"
	"context"
	"errors"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestWizardFollowsLogsOnlyAfterSuccessfulDeploy(t *testing.T) {
	for _, success := range []bool{true, false} {
		t.Run(map[bool]string{true: "success", false: "failure"}[success], func(t *testing.T) {
			d, deployments, _ := testWizardDeps(t, textFixture())
			original := d.deploy
			followed := 0
			d.deploy = func(ctx context.Context, p Plan, wait time.Duration) error {
				if !success {
					return errors.New("fixture startup failure")
				}
				return original(ctx, p, wait)
			}
			d.followLogs = func(_ context.Context, backend, name string) error {
				followed++
				if *deployments != 1 || backend != "native" || name != "modelship" {
					t.Fatal("logs did not follow exact successful deployment")
				}
				return nil
			}
			var out bytes.Buffer
			err := runWizard(context.Background(), strings.NewReader(textWizardInput("y")), &out, d)
			if success && (err != nil || followed != 1 || !strings.Contains(out.String(), "Ctrl-C to exit")) {
				t.Fatal(err, followed, out.String())
			}
			if !success && (err == nil || followed != 0 || strings.Contains(out.String(), "Live logs")) {
				t.Fatal("logs opened after deployment failure")
			}
		})
	}
	d, _, _ := testWizardDeps(t, textFixture())
	d.followLogs = func(context.Context, string, string) error { t.Fatal("logs before approval"); return nil }
	_ = runWizard(context.Background(), strings.NewReader(textWizardInput("n")), &bytes.Buffer{}, d)
}

func TestWizardLogViewerIsReadOnlyAndCancellationIsSuccess(t *testing.T) {
	for _, backend := range []string{"native", "docker"} {
		for _, mode := range []string{"complete", "error", "cancel", "already-canceled"} {
			t.Run(backend+"/"+mode, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				calls := 0
				d := wizardDeps{executable: "/path with spaces/modelship", followLogs: func(_ context.Context, b, n string) error {
					calls++
					if b != backend || n != "fixture-name" {
						t.Fatal("wrong log target")
					}
					if mode == "cancel" {
						cancel()
						return errors.New("signal: killed")
					}
					if mode == "error" {
						return errors.New("fixture-secret-must-not-echo")
					}
					return nil
				}}
				if mode == "already-canceled" {
					cancel()
				}
				var out bytes.Buffer
				wizardFollowLogs(ctx, &out, d, backend, "fixture-name")
				expected := 1
				if mode == "already-canceled" {
					expected = 0
				}
				if calls != expected || !strings.Contains(out.String(), "--backend "+backend+" --name fixture-name") || !strings.Contains(out.String(), "No stop request") && !strings.Contains(out.String(), "no stop request") {
					t.Fatal(calls, out.String())
				}
				if strings.Contains(out.String(), "fixture-secret") || strings.Contains(out.String(), "signal: killed") {
					t.Fatal("subprocess error echoed")
				}
			})
		}
	}
}

func TestWizardZeroWaitStillFollowsLogs(t *testing.T) {
	d, _, _ := testWizardDeps(t, textFixture())
	original := d.deploy
	d.deploy = func(ctx context.Context, p Plan, wait time.Duration) error {
		if wait != 0 {
			t.Fatal("changed wait budget")
		}
		return original(ctx, p, wait)
	}
	followed := 0
	d.followLogs = func(context.Context, string, string) error { followed++; return nil }
	input := strings.Join([]string{"org/model", "", "", "", "", "", "", "", "", "0", "y"}, "\n") + "\n"
	if err := runWizard(context.Background(), strings.NewReader(input), &bytes.Buffer{}, d); err != nil || followed != 1 {
		t.Fatal(err, followed)
	}
}

func TestWizardLogViewerChildInterruptBeforeContextCancellation(t *testing.T) {
	for _, script := range []string{"kill -INT $$", "exit 130"} {
		err := exec.Command("sh", "-c", script).Run()
		if err == nil || !wizardLogInterrupted(context.Background(), err) {
			t.Fatal("missed interrupted viewer", err)
		}
	}
	if wizardLogInterrupted(context.Background(), errors.New("fixture failure")) {
		t.Fatal("ordinary viewer error treated as cancellation")
	}
}
