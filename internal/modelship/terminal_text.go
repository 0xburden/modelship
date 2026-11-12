package modelship

import (
	"fmt"
	"strings"
	"unicode"
)

// Hub filenames/warnings are data, not terminal escape/control instructions.
func terminalText(value string) string {
	var out strings.Builder
	for _, r := range value {
		if unicode.IsControl(r) || unicode.Is(unicode.Cf, r) {
			fmt.Fprintf(&out, "\\u%04x", r)
		} else {
			out.WriteRune(r)
		}
	}
	return out.String()
}
