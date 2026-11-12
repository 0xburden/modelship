#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
: "${VERSION:?Set VERSION, for example v0.1.0}"
case "$VERSION" in v[0-9]* ) ;; *) echo "VERSION must start with v and a digit" >&2; exit 1;; esac
mkdir -p dist
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -buildvcs=false -trimpath -ldflags "-s -w -X main.version=$VERSION" -o dist/modelship_linux_amd64 ./cmd/modelship
cp install.sh dist/install.sh
(cd dist && if command -v sha256sum >/dev/null 2>&1; then sha256sum modelship_linux_amd64 install.sh > SHA256SUMS; else shasum -a 256 modelship_linux_amd64 install.sh > SHA256SUMS; fi)
