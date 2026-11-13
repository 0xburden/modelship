#!/bin/sh
# Build a versioned release site, retaining verified bytes from prior releases.
set -eu
cd "$(dirname "$0")/.."
version=$(cat releases/current)
printf '%s\n' "$version" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9][A-Za-z0-9.-]*)?$' || {
  echo 'Invalid releases/current version' >&2; exit 1;
}
base=${RELEASE_ARCHIVE_BASE:-https://modelship.bubuntu.xyz}
case "$base" in https://*) ;; *) echo 'Release archive must use HTTPS' >&2; exit 1;; esac
publish=${MODELSHIP_PUBLISH_ORIGIN:-$base}
case "$publish" in https://*) ;; *) echo 'Publish origin must use HTTPS' >&2; exit 1;; esac
if [ -n "${RELEASE_VERSION:-}" ]; then
  echo "RELEASE_VERSION build argument is deprecated and ignored; releases/current selects $version." >&2
fi
VERSION="$version" sh scripts/release.sh
stage=$(mktemp -d "$PWD/dist/.release-site.XXXXXX")
trap 'rm -rf "$stage"' EXIT HUP INT TERM
fetch() {
  curl --silent --show-error --location --retry 3 --connect-timeout 15 --max-time 120 \
    --proto '=https' --proto-redir '=https' --tlsv1.2 --header 'Cache-Control: no-cache' "$@"
}
# Refuse to publish different bytes at a version that already exists.
status=$(fetch --output "$stage/published.sha256" --write-out '%{http_code}' "${publish%/}/$version/SHA256SUMS")
case "$status" in
  200) cmp -s dist/SHA256SUMS "$stage/published.sha256" || {
    echo "Refusing to overwrite published $version; bump releases/current." >&2; exit 1;
  } ;;
  404) ;;
  *) echo "Cannot establish release immutability (HTTP $status); refusing publication." >&2; exit 1 ;;
esac
rm "$stage/published.sha256"
# The manifest pins all retained assets, including their checksum file itself.
while read -r digest path; do
  [ -n "$digest" ] || continue
  printf '%s\n' "$path" | grep -Eq '^v[0-9][A-Za-z0-9.-]*/(modelship_linux_amd64|install.sh|SHA256SUMS)$' || {
    echo 'Invalid retained release path' >&2; exit 1;
  }
  case "$path" in "$version"/*) echo 'Current version also listed as retained' >&2; exit 1;; esac
  mkdir -p "$stage/$(dirname "$path")"
  fetch --fail --output "$stage/$path" "${base%/}/$path"
done < releases/retained.sha256
manifest="$PWD/releases/retained.sha256"
if [ -s "$manifest" ]; then
  (cd "$stage" && if command -v sha256sum >/dev/null 2>&1; then sha256sum -c "$manifest"; else shasum -a 256 -c "$manifest"; fi)
fi
mkdir -p "$stage/$version"
cp dist/modelship_linux_amd64 dist/install.sh dist/SHA256SUMS "$stage/$version/"
# Only this entry point is mutable. Copy the already version-baked installer,
# never rebake it against /latest: binary/checksum downloads stay in the same
# immutable version even if a newer release is published during installation.
mkdir -p "$stage/latest"
cp "$stage/$version/install.sh" "$stage/latest/install.sh"
# Do not mix with a prior generated site's contents.
[ ! -e dist/site ] || { echo 'dist/site already exists; use a fresh build directory' >&2; exit 1; }
mv "$stage" dist/site
trap - EXIT HUP INT TERM
printf 'Release site ready: %s (prior release bytes retained)\n' "$version"
