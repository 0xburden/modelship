package modelship

import (
	"context"
	"encoding/binary"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// A real child process with real pipes; no CUDA, weights or network egress.
func TestDiffusionChildFixtureProcess(t *testing.T) {
	mode := os.Getenv("MODELSHIP_DIFFUSION_TEST_CHILD")
	if mode == "" {
		return
	}
	if mode == "startup-hang" {
		time.Sleep(time.Minute)
		os.Exit(1)
	}
	if _, err := os.Stdout.Write(diffusionFixtureFrame('L', 1, nil)); err != nil {
		os.Exit(2)
	}
	for i := 0; ; i++ {
		var header [4]byte
		if _, err := io.ReadFull(os.Stdin, header[:]); err != nil {
			os.Exit(0)
		}
		n := binary.BigEndian.Uint32(header[:])
		if n < 5 || n > 1028 {
			os.Exit(3)
		}
		b := make([]byte, n)
		if _, err := io.ReadFull(os.Stdin, b); err != nil {
			os.Exit(4)
		}
		if i == 0 {
			if binary.BigEndian.Uint32(b[:4]) != diffusionWarmupSeed || string(b[4:]) != diffusionWarmupPrompt {
				os.Exit(5)
			}
			if mode == "warmup-fail" {
				_, _ = os.Stdout.Write(diffusionFixtureFrame('F', 0, nil))
				time.Sleep(time.Minute)
				os.Exit(6)
			}
			if mode == "gated" {
				for deadline := time.Now().Add(8 * time.Second); ; time.Sleep(10 * time.Millisecond) {
					if _, err := os.Stat(os.Getenv("MODELSHIP_DIFFUSION_TEST_GATE")); err == nil {
						break
					}
					if time.Now().After(deadline) {
						os.Exit(7)
					}
				}
			}
		} else if mode == "request-hang" {
			time.Sleep(time.Minute)
			os.Exit(8)
		}
		if _, err := os.Stdout.Write(diffusionFixtureFrame('I', 3, make([]byte, 3*diffusionSide*diffusionSide))); err != nil {
			os.Exit(9)
		}
	}
}

func TestDiffusionSupervisor(t *testing.T) {
	for _, mode := range []string{"gated", "startup-hang", "warmup-fail", "request-hang"} {
		t.Run(mode, func(t *testing.T) {
			listener, err := net.Listen("tcp4", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			defer listener.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			gate := t.TempDir() + "/release-warmup"
			cmd := exec.Command(os.Args[0], "-test.run=^TestDiffusionChildFixtureProcess$")
			cmd.Env = append(os.Environ(), "MODELSHIP_DIFFUSION_TEST_CHILD="+mode, "MODELSHIP_DIFFUSION_TEST_GATE="+gate)
			startup, request := 8*time.Second, 4*time.Second
			if mode == "startup-hang" {
				startup = 300 * time.Millisecond
			}
			if mode == "request-hang" {
				request = 250 * time.Millisecond
			}
			result := make(chan error, 1)
			go func() {
				result <- serveDiffusion(ctx, listener, cmd, "test-model", strings.Repeat("a", 40), strings.Repeat("b", 32), diffusionFixtureKey, startup, request)
			}()
			client := nativeClient(5 * time.Second)
			defer client.CloseIdleConnections()
			call := func(method, path, body string) (int, error) {
				req, err := http.NewRequest(method, "http://"+listener.Addr().String()+path, strings.NewReader(body))
				if err != nil {
					return 0, err
				}
				req.Header.Set("Authorization", "Bearer "+diffusionFixtureKey)
				resp, err := client.Do(req)
				if err != nil {
					return 0, err
				}
				defer resp.Body.Close()
				_, err = io.Copy(io.Discard, resp.Body)
				return resp.StatusCode, err
			}
			if mode == "gated" {
				if code, err := call("GET", "/health", ""); err != nil || code != 503 {
					t.Fatalf("ready without a completed synchronized warm-up: %d %v", code, err)
				}
				if err := os.WriteFile(gate, nil, 0600); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "gated" || mode == "request-hang" {
				for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(20 * time.Millisecond) {
					code, err := call("GET", "/health", "")
					if err == nil && code == 200 {
						break
					}
					if time.Now().After(deadline) {
						t.Fatalf("did not become ready: %d %v", code, err)
					}
				}
				code, err := call("POST", "/v1/images/generations", `{"prompt":"a lake","seed":0}`)
				if mode == "gated" {
					if err != nil || code != 200 {
						t.Fatalf("framed generation: %d %v", code, err)
					}
					cancel()
				} else if err == nil && code != 503 {
					t.Fatalf("uncertain response returned success: %d", code)
				}
			}
			select {
			case err := <-result:
				if (err == nil) != (mode == "gated") || cmd.ProcessState == nil {
					t.Fatalf("wrong exit or child was not reaped: %v", err)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("supervisor failed to terminate/reap")
			}
		})
	}
}
