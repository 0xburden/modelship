#!/bin/sh
# Configure this immutable release URL when publishing the bootstrap.
set -eu
: "${MODELSHIP_RELEASE_BASE:?Set MODELSHIP_RELEASE_BASE to an HTTPS release directory (see README)}"
case "$MODELSHIP_RELEASE_BASE" in https://*) ;; *) echo "Release base must use HTTPS" >&2; exit 1;; esac
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) asset=modelship_linux_amd64 ;;
  *) echo "This MVP supports Linux x86_64 NVIDIA servers only." >&2; exit 1 ;;
esac
command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
base=${MODELSHIP_RELEASE_BASE%/}
fetch() { curl --fail --silent --show-error --location --retry 3 --proto '=https' --proto-redir '=https' --tlsv1.2 "$1" -o "$2"; }
fetch "$base/$asset" "$tmp/$asset"
fetch "$base/SHA256SUMS" "$tmp/SHA256SUMS"
expected=$(awk -v file="$asset" '$2 == file { print $1 }' "$tmp/SHA256SUMS")
if [ "${#expected}" -ne 64 ] || printf '%s' "$expected" | grep -q '[^a-fA-F0-9]'; then
  echo "Missing or invalid SHA256 checksum" >&2; exit 1
fi
if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum "$tmp/$asset" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
  actual=$(shasum -a 256 "$tmp/$asset" | awk '{print $1}')
else
  echo "sha256sum or shasum is required" >&2; exit 1
fi
[ "$expected" = "$actual" ] || { echo "Checksum mismatch; refusing installation" >&2; exit 1; }
dest=${MODELSHIP_INSTALL_DIR:-"$HOME/.local/bin"}
mkdir -p "$dest"
# Same-directory rename prevents leaving a partially copied binary.
stage=$(mktemp "$dest/.modelship.XXXXXX")
if ! cp "$tmp/$asset" "$stage" || ! chmod 0755 "$stage" || ! mv -f "$stage" "$dest/modelship"; then
  rm -f "$stage"; echo "Installation failed" >&2; exit 1
fi
echo "Installed $dest/modelship. Add $dest to PATH if needed." >&2
if [ "$#" -gt 0 ]; then
  exec "$dest/modelship" "$@"
fi
"$dest/modelship" version
