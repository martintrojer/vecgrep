# vecgrep TUI demo
# Record with: vhs doc/demo.tape
# Requires: vecgrep on PATH, nvim, vhs installed (brew install vhs)

Output doc/demo.gif
Set Width 1200
Set Height 700
Set FontSize 16
Set Shell zsh
Set Theme "Catppuccin Mocha"
Set TypingSpeed 50ms

Type "vecgrep --clear-cache -l 'creates index, first time only'"
Enter
Sleep 6s

# --- grep can't find it, vecgrep can ---
Type "echo '--- normal grep matches exact text, not meaning (thus finds nothing in this example) ---'"
Enter
Sleep 1s
Type "grep -rn 'handle errors' src/"
Enter
Sleep 1500ms

Type "echo '--- vecgrep searches by meaning (finds relevant parts of files) ---'"
Enter
Sleep 500ms
Type "vecgrep 'handle errors' src/"
Enter
Sleep 3s

# --- Interactive TUI with live search ---
Type "echo '--- interactive mode: search, browse, open in editor ---'"
Enter
Sleep 500ms
Type "vecgrep -i 'cache invalidation'"
Enter
Sleep 2s

# Navigate results — preview follows selection
Down
Sleep 800ms
Down
Sleep 800ms
Down
Sleep 1500ms

# Live re-search with a different query
Backspace 20
Type "how files are discovered"
Sleep 3s

# Browse new results
Down
Sleep 800ms
Down
Sleep 1500ms

# Toggle preview off and back
Tab
Sleep 1200ms
Tab
Sleep 1500ms

# --- Open result in neovim ---
Enter
Sleep 2s
Up
Sleep 100ms
Down
Sleep 100ms
Down
Sleep 1s

# Exit neovim
Type ":q"
Enter
Sleep 1s

# --- CLI one-liner: list matching files ---
Type "echo '--- also works as a CLI tool, showing matching hunks or files ---'"
Enter
Sleep 500ms
Type "vecgrep -ql 'database schema' src/"
Enter
Sleep 3s
