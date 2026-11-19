package modelship

import (
	"context"
	"fmt"
	"image"
	"net"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"
)

// Only the verified installer/entry point may construct a production
// command: absolute pinned runtime and bundle, a curated environment, inherited
// per-GPU lock in ExtraFiles, and fresh device admission. No shell or child HTTP.
// The child stays in the native supervisor's process group so stop can reap both.
func serveDiffusion(ctx context.Context, listener net.Listener, cmd *exec.Cmd, task, model, revision, runID, key string, startupTimeout, requestTimeout time.Duration) error {
	defer listener.Close()
	if startupTimeout <= 0 || requestTimeout <= 0 {
		return fmt.Errorf("invalid diffusion time budgets")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	// Use explicit os.Pipes so Wait cannot close the reader behind a frame
	// reader, and SetDeadline can bound reads and writes without IO goroutines.
	childIn, toChild, err := os.Pipe()
	if err != nil {
		return fmt.Errorf("could not create diffusion request pipe")
	}
	defer childIn.Close()
	defer toChild.Close()
	fromChild, childOut, err := os.Pipe()
	if err != nil {
		return fmt.Errorf("could not create diffusion response pipe")
	}
	defer fromChild.Close()
	defer childOut.Close()
	wire := diffusionWire{in: fromChild, out: toChild}
	deadlines := func(limit time.Duration) error {
		deadline := time.Now().Add(limit)
		if fromChild.SetReadDeadline(deadline) != nil || toChild.SetWriteDeadline(deadline) != nil {
			return errDiffusionProtocol
		}
		return nil
	}
	if err := deadlines(startupTimeout); err != nil {
		return err
	}
	exited := make(chan struct{})
	var terminate sync.Once
	kill := func() {
		terminate.Do(func() {
			// Interrupt pending protocol IO independent of the HTTP client.
			_ = cmd.Process.Kill()
			_ = toChild.Close()
			_ = fromChild.Close()
		})
	}
	gateway, err := newDiffusionGatewayForTask(task, model, revision, runID, key, func(prompt string, seed uint32, rgb []byte) (*image.NRGBA, error) {
		if err := deadlines(requestTimeout); err != nil {
			return nil, err
		}
		if task == diffusionEditTask {
			return wire.edit(prompt, seed, rgb)
		}
		return wire.generate(prompt, seed)
	}, kill)
	if err != nil {
		return err
	}
	// Don't mirror untrusted upstream diagnostics (which can contain prompts).
	cmd.Stdin, cmd.Stdout, cmd.Stderr = childIn, childOut, nil
	release := guardDiffusionChild(cmd)
	defer release()
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("verified private diffusion worker failed to launch")
	}
	_ = childIn.Close()
	_ = childOut.Close()
	go func() {
		_ = cmd.Wait()
		gateway.unavailable()
		close(exited)
	}()

	server := &http.Server{Handler: gateway, ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 30 * time.Second,
		WriteTimeout: requestTimeout + 45*time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16 << 10}
	served := make(chan error, 1)
	go func() { served <- server.Serve(listener) }()
	warmup := make(chan error, 1)
	go func() {
		// L proves successful construction only. Require a synchronized
		// response for the selected task in this very same child before ready.
		if task == diffusionEditTask {
			if err := wire.loadedVersion(diffusionEditProtocol); err != nil {
				warmup <- err
				return
			}
			_, err := wire.edit(diffusionEditWarmupPrompt, diffusionWarmupSeed, diffusionEditWarmupReference())
			warmup <- err
			return
		}
		if err := wire.loaded(); err != nil {
			warmup <- err
			return
		}
		_, err := wire.generate(diffusionWarmupPrompt, diffusionWarmupSeed)
		warmup <- err
	}()

	var result error
loop:
	for {
		select {
		case <-ctx.Done():
			break loop
		case <-exited:
			if ctx.Err() == nil {
				result = fmt.Errorf("private diffusion worker exited; check readiness before restarting")
			}
			break loop
		case <-served:
			if ctx.Err() == nil {
				result = fmt.Errorf("diffusion HTTP listener stopped")
			}
			break loop
		case err := <-warmup:
			warmup = nil
			if err != nil {
				result = fmt.Errorf("diffusion startup or synchronized %s probe failed", task)
				break loop
			}
			gateway.warmed()
		}
	}
	gateway.unavailable()
	// No further request can acquire a ready GPU. Kill only our child and
	// reap before returning (including failed startup and signal shutdown).
	_ = server.Close()
	kill()
	<-exited
	if warmup != nil {
		<-warmup
	}
	return result
}
