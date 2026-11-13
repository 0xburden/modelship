package modelship

import (
	"runtime"
	"testing"
)

func TestGGUFActualLinuxELFABIInspection(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Linux system ELF libraries required")
	}
	have, e := inspectRuntimeABI()
	if e != nil {
		t.Fatal(e)
	}
	if !abiVersionPattern.MatchString(have.GLIBC) || !abiVersionPattern.MatchString(have.GLIBCXX) {
		t.Fatal("invalid native ABI inspection", have)
	}
	t.Logf("Actual native ELF symbols: GLIBC_%s GLIBCXX_%s", have.GLIBC, have.GLIBCXX)
	if compareABIVersion(have.GLIBC, "2.35") < 0 || compareABIVersion(have.GLIBCXX, "3.4.30") < 0 {
		t.Skip("test host is older than the repaired runtime baseline")
	}
	need := runtimeABI{GLIBC: "2.35", GLIBCXX: "3.4.30"}
	if e = validateRuntimeABI(have, need); e != nil {
		t.Fatal(e)
	}
}
