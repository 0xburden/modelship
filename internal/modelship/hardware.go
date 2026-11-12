package modelship

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

type GPU struct {
	Index    int    `json:"index"`
	UUID     string `json:"uuid"`
	Name     string `json:"name"`
	TotalMiB int64  `json:"total_mib"`
	FreeMiB  int64  `json:"free_mib"`
	Driver   string `json:"driver"`
	Compute  string `json:"compute_capability"`
}
type Host struct {
	OS                string   `json:"os"`
	Arch              string   `json:"arch"`
	GPUs              []GPU    `json:"gpus"`
	Docker            bool     `json:"docker_available"`
	CPUs              int      `json:"cpu_count"`
	RAMAvailableMiB   int64    `json:"ram_available_mib,omitempty"`
	DockerDiskFreeMiB int64    `json:"docker_disk_free_mib,omitempty"`
	DockerEndpoint    string   `json:"docker_endpoint,omitempty"`
	Warnings          []string `json:"warnings,omitempty"`
}

func ParseGPUs(output string) ([]GPU, error) {
	records, err := csv.NewReader(strings.NewReader(strings.TrimSpace(output))).ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parse nvidia-smi: %w", err)
	}
	var gpus []GPU
	for _, row := range records {
		if len(row) != 7 {
			return nil, fmt.Errorf("unexpected nvidia-smi columns: got %d, want 7", len(row))
		}
		for i := range row {
			row[i] = strings.TrimSpace(row[i])
		}
		idx, e1 := strconv.Atoi(row[0])
		total, e2 := strconv.ParseInt(row[3], 10, 64)
		free, e3 := strconv.ParseInt(row[4], 10, 64)
		if e1 != nil || e2 != nil || e3 != nil || idx < 0 || total <= 0 || free < 0 || free > total || !strings.HasPrefix(row[1], "GPU-") {
			return nil, fmt.Errorf("invalid GPU index, UUID, or memory from nvidia-smi")
		}
		gpus = append(gpus, GPU{Index: idx, UUID: row[1], Name: row[2], TotalMiB: total, FreeMiB: free, Driver: row[5], Compute: row[6]})
	}
	if len(gpus) == 0 {
		return nil, fmt.Errorf("no NVIDIA GPUs found")
	}
	return gpus, nil
}
func Detect(ctx context.Context) (Host, error) { return DetectForBackend(ctx, "docker") }
func DetectForBackend(ctx context.Context, backend string) (Host, error) {
	h := Host{OS: runtime.GOOS, Arch: runtime.GOARCH, CPUs: runtime.NumCPU()}
	if data, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			fields := strings.Fields(line)
			if len(fields) >= 2 && fields[0] == "MemAvailable:" {
				kb, _ := strconv.ParseInt(fields[1], 10, 64)
				h.RAMAvailableMiB = kb / 1024
			}
		}
	}
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version,compute_cap", "--format=csv,noheader,nounits").CombinedOutput()
	if err != nil {
		return h, fmt.Errorf("NVIDIA discovery failed: install a working NVIDIA driver and ensure nvidia-smi is available")
	}
	h.GPUs, err = ParseGPUs(string(out))
	if err != nil {
		return h, err
	}
	if backend == "native" {
		if h.OS != "linux" || h.Arch != "amd64" {
			h.Warnings = append(h.Warnings, "deployment MVP supports Linux amd64 only")
		}
		return h, nil
	}
	root, dockerErr := output(ctx, "info", "--format", "{{.DockerRootDir}}")
	h.Docker = dockerErr == nil
	if h.Docker {
		h.DockerEndpoint, _ = output(ctx, "context", "inspect", "--format", "{{.Endpoints.docker.Host}}")
		if endpoint := os.Getenv("DOCKER_HOST"); endpoint != "" {
			h.DockerEndpoint = endpoint
		}
		if data, err := exec.CommandContext(ctx, "df", "-Pk", root).Output(); err == nil {
			lines := strings.Split(strings.TrimSpace(string(data)), "\n")
			fields := strings.Fields(lines[len(lines)-1])
			if len(fields) >= 4 {
				kb, _ := strconv.ParseInt(fields[3], 10, 64)
				h.DockerDiskFreeMiB = kb / 1024
			}
		}
		if h.DockerDiskFreeMiB == 0 {
			h.Warnings = append(h.Warnings, "Docker filesystem free space could not be inspected; check disk capacity manually")
		}
	}
	if !h.Docker {
		h.Warnings = append(h.Warnings, "Docker daemon unavailable: check Docker installation and user permissions; GPU access must use your provider's supported NVIDIA/CDI setup")
	}
	if h.OS != "linux" || h.Arch != "amd64" {
		h.Warnings = append(h.Warnings, "deployment MVP supports Linux amd64 only")
	}
	if os.Getenv("DOCKER_HOST") != "" || os.Getenv("DOCKER_CONTEXT") != "" {
		h.Warnings = append(h.Warnings, "Docker context must target this NVIDIA host; remote Docker hosts are unsupported")
	}
	return h, nil
}
func driverAtLeast(got string, major, minor int) bool {
	p := strings.Split(got, ".")
	if len(p) < 2 {
		return false
	}
	a, e1 := strconv.Atoi(p[0])
	b, e2 := strconv.Atoi(p[1])
	return e1 == nil && e2 == nil && (a > major || (a == major && b >= minor))
}
