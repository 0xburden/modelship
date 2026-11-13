package modelship

import (
	"debug/elf"
	"fmt"
	"strconv"
	"strings"
)

type runtimeABI struct{ GLIBC, GLIBCXX string }

func compareABIVersion(a, b string) int {
	aa, bb := strings.Split(a, "."), strings.Split(b, ".")
	for i := 0; i < max(len(aa), len(bb)); i++ {
		av, bv := 0, 0
		if i < len(aa) {
			av, _ = strconv.Atoi(aa[i])
		}
		if i < len(bb) {
			bv, _ = strconv.Atoi(bb[i])
		}
		if av < bv {
			return -1
		}
		if av > bv {
			return 1
		}
	}
	return 0
}
func maxABISymbol(names []string, prefix string) string {
	best := ""
	for _, name := range names {
		v, ok := strings.CutPrefix(name, prefix+"_")
		if !ok || v == "" {
			continue
		}
		valid := true
		for _, p := range strings.Split(v, ".") {
			if p == "" {
				valid = false
				break
			}
			for _, c := range p {
				if c < '0' || c > '9' {
					valid = false
					break
				}
			}
		}
		if valid && (best == "" || compareABIVersion(v, best) > 0) {
			best = v
		}
	}
	return best
}
func libraryABIVersion(paths []string, prefix string) (string, error) {
	for _, path := range paths {
		f, e := elf.Open(path)
		if e != nil {
			continue
		}
		symbols, e := f.DynamicSymbols()
		f.Close()
		if e != nil {
			continue
		}
		names := make([]string, 0, len(symbols))
		for _, s := range symbols {
			names = append(names, s.Name)
		}
		if version := maxABISymbol(names, prefix); version != "" {
			return version, nil
		}
	}
	return "", fmt.Errorf("cannot inspect system %s symbol versions; a compatible glibc-based Linux userspace is required", prefix)
}
func inspectRuntimeABI() (runtimeABI, error) {
	var abi runtimeABI
	var e error
	abi.GLIBC, e = libraryABIVersion([]string{"/lib/x86_64-linux-gnu/libc.so.6", "/usr/lib/x86_64-linux-gnu/libc.so.6", "/lib64/libc.so.6", "/usr/lib64/libc.so.6"}, "GLIBC")
	if e != nil {
		return abi, e
	}
	abi.GLIBCXX, e = libraryABIVersion([]string{"/usr/lib/x86_64-linux-gnu/libstdc++.so.6", "/lib/x86_64-linux-gnu/libstdc++.so.6", "/usr/lib64/libstdc++.so.6", "/lib64/libstdc++.so.6"}, "GLIBCXX")
	return abi, e
}
func validateRuntimeABI(have, need runtimeABI) error {
	if have.GLIBC == "" || have.GLIBCXX == "" {
		return fmt.Errorf("system glibc/libstdc++ ABI could not be verified")
	}
	if compareABIVersion(have.GLIBC, need.GLIBC) < 0 || compareABIVersion(have.GLIBCXX, need.GLIBCXX) < 0 {
		return fmt.Errorf("llama.cpp runtime requires GLIBC_%s and GLIBCXX_%s; system provides GLIBC_%s and GLIBCXX_%s. No host libraries were changed; use a compatible Modelship runtime, not a glibc upgrade or host LD_LIBRARY_PATH override", need.GLIBC, need.GLIBCXX, have.GLIBC, have.GLIBCXX)
	}
	return nil
}
func checkLlamaRuntimeABI(p Plan) error {
	if p.GGUF == nil {
		return fmt.Errorf("missing GGUF runtime specification")
	}
	spec, e := loadLlamaRuntime(p.GGUF.Runtime)
	if e != nil {
		return e
	}
	have, e := inspectRuntimeABI()
	if e != nil {
		return e
	}
	return validateRuntimeABI(have, runtimeABI{GLIBC: spec.MinimumGLIBC, GLIBCXX: spec.MinimumGLIBCXX})
}
