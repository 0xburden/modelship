package modelship

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"syscall"
)

// Deployment has already succeeded. Following logs is observational: failure or
// cancellation must never turn success into a retry, restart, or stop request.
func wizardFollowLogs(ctx context.Context, out io.Writer, d wizardDeps, backend, name string) {
	command := fmt.Sprintf("%s logs --backend %s --name %s", shellWord(d.executable), shellWord(backend), shellWord(name))
	fmt.Fprintf(out, "Reopen logs: %s\n", command)
	if d.followLogs == nil {
		return
	} // CPU fixtures may omit this side effect.
	if ctx.Err() != nil {
		fmt.Fprintln(out, "Log viewer not opened: session interrupted. No stop request was sent.")
		return
	}
	fmt.Fprintln(out, "\nLive logs — Ctrl-C to exit the log viewer without stopping the worker.")
	err := d.followLogs(ctx, backend, name)
	switch {
	case wizardLogInterrupted(ctx, err):
		fmt.Fprintln(out, "\nExited log viewer. No stop request was sent.")
	case err != nil:
		// Do not echo arbitrary subprocess errors or imply deployment failed.
		fmt.Fprintf(out, "\nLog viewer ended unexpectedly; deployment was submitted, and no stop request was sent. Reopen with: %s\n", command)
	default:
		fmt.Fprintln(out, "\nLog stream ended. No stop request was sent; use the status command above to check the worker.")
	}
}

// A terminal SIGINT reaches both the viewer child and the CLI. The child may
// exit before signal.NotifyContext has published cancellation to the parent.
func wizardLogInterrupted(ctx context.Context, err error) bool {
	if ctx.Err() != nil || errors.Is(err, context.Canceled) {
		return true
	}
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		if status, ok := exit.Sys().(syscall.WaitStatus); ok && status.Signaled() {
			return status.Signal() == syscall.SIGINT || status.Signal() == syscall.SIGTERM
		}
		return exit.ExitCode() == 130
	}
	return false
}
