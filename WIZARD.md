# Interactive terminal selectors (alpha.12 candidate)

The verified CLI—not the shell installer—renders fixed-choice radio-style menus. Normal SSH with a TTY is supported; no desktop, mouse, special fonts or new UI dependency is needed.

```text
Input mode (number, letter, or full value)
Options 1-2 of 2
> (o) 1 / a) text-only (default)
  ( ) 2 / b) vision-images
Up/Down: select  Enter: confirm  Esc: back
Shortcut (number/letter/value):
```

- **Up/Down** changes the highlighted option; **Enter** submits it. Navigation stops at the first/last option rather than wrapping. A missing default requires an explicit selection.
- **Numbers, letters and full values** still work, case-insensitively; multi-digit numbers need Enter. Typing a shortcut overrides the highlight; an arrow clears the typed shortcut and resumes selection.
- Long lists (including the reviewed GGUF quantizations) scroll in a bounded viewport with option indexes. Long display labels are clipped safely; the complete canonical selected value is printed after confirmation and preserved in history/the reviewed plan. Control/Unicode formatting characters cannot inject terminal commands; menu display is ASCII for predictable width.
- **Esc** or typed **`:back` + Enter** edits the preceding answer. Its previous option is highlighted, but later selections/approvals are discarded. Immutable metadata may be reused; live GPU/cache/fit checks still refresh. Prior safety approvals are never restored as the default when revisited.
- **Ctrl-C**, EOF and errors cancel before deployment approval. Terminal settings are restored on exit from every prompt; no hidden cursor or alternate-screen state is used. Stop/cancel after deployment approval retains the existing worker lifecycle semantics.

Backend, GPU, task, input mode, GGUF artifact, API access/key policy and Docker GPU mode use selectors when multiple options are available. A single supported choice still auto-selects, except where an explicit artifact choice was already required. URLs/revisions, names, ports, context, Python/key file paths and durations remain typed.

**Safety consents and final deployment confirmation remain separate, typed y/N prompts, default No.** They are not generic option selectors. Experimental, research, reviewed-code, WAN conversion and heuristic-risk approvals remain independent. No model tensors, installs, deployment state or new keys before final approval; physical placement/disk/security constraints remain hard. No adapters, inference behavior or saved workers change with this UI feature.

## Plain terminals and automation

```sh
modelship launch --plain
# Through a downloaded verified installer:
sh install.sh launch --plain
```

`--plain` retains numbered line menus. Missing/dumb/unknown `TERM`, unavailable window dimensions, widths below 50 columns or heights below 10 rows also fall back automatically. Recognized ANSI terminal families include xterm, screen, tmux, rxvt, vt100/vt220, ansi and linux. If a live selector is resized too small, input cancels safely; restart with `--plain`.

Both input/output must still be terminals for `launch`, even with `--plain`. CI, redirected streams and SSH without a TTY must use the existing flag commands (`deploy MODEL ... --yes`); there is no piped-input deployment wizard. `--help` and `version` never prompt. The installer still downloads/checksums the CLI before invoking the same shared wizard/planner/deployer.

## Validation scope

CPU-only unit and real PTY tests exercise selectors, scrolling/boundaries, default/no-default, numeric/letter/full-value shortcuts, invalid input, Back/replay, canonical history, output sanitation, fallback modes, cancellation and termios restoration. A dependency-injected real wizard test checks metadata reuse, fresh hardware/cache checks and zero key/deployment calls on final No. No GPU/model download/inference or existing worker activity is required or claimed.
