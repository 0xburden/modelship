#!/bin/sh
# Release packaging replaces this marker with an immutable HTTPS URL.
release_default='__MODELSHIP_RELEASE_BASE__'
set -eu
if [ -z "${MODELSHIP_RELEASE_BASE:-}" ]; then
  case "$release_default" in
    https://*) MODELSHIP_RELEASE_BASE=$release_default ;;
    *) echo "Source installer requires MODELSHIP_RELEASE_BASE; published installers already know their versioned URL." >&2; exit 1 ;;
  esac
fi
case "$MODELSHIP_RELEASE_BASE" in https://*) ;; *) echo "Release base must use HTTPS" >&2; exit 1;; esac
case "$release_default" in
  https://*) if [ "$MODELSHIP_RELEASE_BASE" != "$release_default" ]; then
    echo "Using MODELSHIP_RELEASE_BASE override rather than this installer's baked release URL." >&2
  fi ;;
esac
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
# exec does not run EXIT traps: clean the download directory before launching.
rm -rf "$tmp"
trap - EXIT HUP INT TERM
if [ "$#" -gt 0 ]; then
  exec "$dest/modelship" "$@"
fi
if [ -t 0 ] && [ -t 1 ]; then
  # An explicit base override can deliberately install an older CLI. Do not
  # invoke a command that version does not implement.
  if "$dest/modelship" help | grep -q '^  launch[[:space:]]'; then
    exec "$dest/modelship" launch
  fi
  echo "Selected CLI has no guided launch; installed without setup." >&2
fi
"$dest/modelship" version
printf 'Installed without interactive setup. Run %s launch from a terminal, or use deploy MODEL --yes for automation.\n' "$dest/modelship" >&2
