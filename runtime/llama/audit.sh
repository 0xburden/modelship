#!/bin/sh
# Audit ELF version needs without executing any model or replacing host libraries.
set -eu
root=${1:?bundle directory}; report=${2:?report path}; libraries=${3:-$root}
: > "$report"
check_version() {
  ns=$1 max=$2 file=$3
  versions=$(readelf -W --version-info "$file" | grep -oE "${ns}_[0-9]+(\.[0-9]+)*" | sort -Vu || true)
  [ -z "$versions" ] || printf '%s: %s\n' "$(basename "$file")" "$versions" >> "$report"
  highest=$(printf '%s\n%s_%s\n' "$versions" "$ns" "$max" | sort -V | tail -n 1)
  [ "$highest" = "${ns}_$max" ] || { echo "Incompatible ABI requirement $highest in $file (maximum ${ns}_$max)" >&2; exit 1; }
}
find "$root" -type f | sort | while IFS= read -r file; do
  if ! readelf -h "$file" >/dev/null 2>&1; then continue; fi
  readelf -h "$file" | grep -q 'Advanced Micro Devices X86-64' || { echo 'Wrong ELF architecture' >&2; exit 1; }
  case "$(basename "$file")" in libc.so*|libm.so*|libpthread.so*|ld-linux*|libstdc++*|libgcc*|libcuda.so*) echo 'Host C/C++/driver library must not be bundled' >&2; exit 1;; esac
  check_version GLIBC 2.35 "$file"
  check_version GLIBCXX 3.4.29 "$file"
  check_version CXXABI 1.3.13 "$file"
  LD_LIBRARY_PATH="$libraries" ldd "$file" >> "$report" 2>&1 || { echo 'Runtime dependency resolution failed' >&2; exit 1; }
done
if grep -q 'not found' "$report"; then echo 'Unresolved runtime dependency' >&2; exit 1; fi
printf 'ABI_AUDIT_PASSED: GLIBC<=2.35 GLIBCXX<=3.4.29 CXXABI<=1.3.13\n' >> "$report"
