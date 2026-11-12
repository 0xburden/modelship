#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
: "${VERSION:?Set VERSION, for example v0.1.0}"
printf '%s\n' "$VERSION" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9][A-Za-z0-9.-]*)?$' || { echo "Invalid release VERSION" >&2; exit 1; }
origin=${MODELSHIP_PUBLISH_ORIGIN:-https://modelship.bubuntu.xyz}
printf '%s\n' "$origin" | grep -Eq '^https://[A-Za-z0-9][A-Za-z0-9.-]*(:[0-9]+)?$' || { echo "MODELSHIP_PUBLISH_ORIGIN must be an HTTPS origin without path/query/credentials" >&2; exit 1; }
mkdir -p dist
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -buildvcs=false -trimpath -ldflags "-s -w -X main.version=$VERSION" -o dist/modelship_linux_amd64 ./cmd/modelship
sed "s|__MODELSHIP_RELEASE_BASE__|$origin/$VERSION|g" install.sh > dist/install.sh
(cd dist && if command -v sha256sum >/dev/null 2>&1; then sha256sum modelship_linux_amd64 install.sh > SHA256SUMS; else shasum -a 256 modelship_linux_amd64 install.sh > SHA256SUMS; fi)
