package main

import (
	"fmt"
	"modelship/internal/modelship"
	"os"
)

var version = "dev"

func main() {
	if err := modelship.Run(os.Args[1:], version); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
