package modelship

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Exercises the real Go gateway -> separately authenticated child -> lifecycle
// path with a CPU-only synthetic engine. It never bypasses the public CLI's
// pinned-model/GPU gates and is NOT evidence of llama.cpp/model compatibility.
func TestGGUFRealPrivateChildLifecycleCPUFixture(t *testing.T) {
	python, e := exec.LookPath("python3")
	if e != nil {
		t.Skip("python3 required for CPU fixture")
	}
	if _, e := os.Stat("/usr/bin/python3"); e == nil {
		python = "/usr/bin/python3"
	} // No setup-python LD_LIBRARY_PATH dependency in the synthetic child.
	m, h, o := ggufPlanFixture(t)
	p, e := BuildPlan(m, h, o)
	if e != nil {
		t.Fatal(e)
	}
	listener, e := net.Listen("tcp4", "127.0.0.1:0")
	if e != nil {
		t.Fatal(e)
	}
	p.Port = listener.Addr().(*net.TCPAddr).Port
	listener.Close()
	p.APIKeyFile = fixtureAPIKey(t)
	ownerKey, _ := readAPIKey(p.APIKeyFile)
	dir := t.TempDir()
	serverPath := filepath.Join(dir, "runtime", "llama-server")
	if e = os.MkdirAll(filepath.Dir(serverPath), 0700); e != nil {
		t.Fatal(e)
	}
	script := `import sys,os,json,signal
from http.server import HTTPServer,BaseHTTPRequestHandler
args=sys.argv[1:]
def arg(name): return args[args.index(name)+1]
assert arg('--host')=='127.0.0.1'
assert arg('--fit')=='off'
assert arg('--gpu-layers')=='all'
assert arg('--device')=='CUDA0'
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-test'
assert 'HF_TOKEN' not in os.environ
key=open(arg('--api-key-file')).read().strip()
assert key not in ' '.join(args)
open(os.path.join(os.getcwd(),'engine-port'),'w').write(arg('--port'))
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.headers.get('Authorization')!='Bearer '+key:self.send_response(401);self.end_headers();return
  self.send_response(200);self.end_headers();self.wfile.write(b'{"ready":true}')
 def do_POST(self):
  if self.headers.get('Authorization')!='Bearer '+key:self.send_response(401);self.end_headers();return
  self.rfile.read(int(self.headers.get('Content-Length','0')))
  self.send_response(200);self.end_headers();self.wfile.write(b'{"id":"cpu-fixture","choices":[{"message":{"content":"Paris"},"finish_reason":"stop"}]}')
 def log_message(self,*args):pass
signal.signal(signal.SIGINT,lambda *_:sys.exit(0))
signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
HTTPServer(('127.0.0.1',int(arg('--port'))),Handler).serve_forever()
`
	// An absolute shebang does not rely on the caller's environment/PATH. Python
	// is a test fixture only; the production adapter does not install/use Python.
	if e = os.WriteFile(serverPath, []byte("#!"+python+"\n"+script), 0700); e != nil {
		t.Fatal(e)
	}
	cfg := NativeConfig{Plan: p, RuntimeBinary: "runtime/llama-server"}
	t.Setenv("HF_TOKEN", "must-not-reach-private-child")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- serveGGUFWorker(ctx, cfg, dir, "cpu-fixture-run") }()
	defer func() {
		cancel()
		select {
		case e := <-done:
			if e != nil {
				t.Error(e)
			}
		case <-time.After(8 * time.Second):
			t.Error("child/gateway did not shut down")
		}
	}()
	client := nativeClient(2 * time.Second)
	defer client.CloseIdleConnections()
	url := fmt.Sprintf("http://127.0.0.1:%d", p.Port)
	deadline := time.Now().Add(8 * time.Second)
	ready := false
	for time.Now().Before(deadline) {
		req, _ := http.NewRequest("GET", url+"/health", nil)
		req.Header.Set("Authorization", "Bearer "+ownerKey)
		resp, e := client.Do(req)
		if e == nil {
			data, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if resp.StatusCode == 200 && strings.Contains(string(data), `"ready":true`) {
				ready = true
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !ready {
		t.Fatal("CPU fixture did not become ready")
	}
	resp, e := client.Get(url + "/health")
	if e != nil {
		t.Fatal(e)
	}
	resp.Body.Close()
	if resp.StatusCode != 401 {
		t.Fatal("public owner authentication bypass")
	}
	req, _ := http.NewRequest("POST", url+"/v1/completions", strings.NewReader(`{"prompt":"Capital of France?"}`))
	req.Header.Set("Authorization", "Bearer "+ownerKey)
	resp, e = client.Do(req)
	if e != nil {
		t.Fatal(e)
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || !strings.Contains(string(data), "Paris") {
		t.Fatal("public-to-private fixture generation failed", resp.StatusCode, string(data))
	}
	raw, e := os.ReadFile(filepath.Join(dir, "engine-port"))
	if e != nil {
		t.Fatal(e)
	}
	privatePort, e := strconv.Atoi(string(raw))
	if e != nil {
		t.Fatal(e)
	}
	unauth, e := client.Get(fmt.Sprintf("http://127.0.0.1:%d/health", privatePort))
	if e != nil {
		t.Fatal(e)
	}
	unauth.Body.Close()
	if unauth.StatusCode != 401 {
		t.Fatal("private backend lacks its own independent key")
	}
}
