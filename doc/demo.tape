# vecgrep TUI demo
# Record with: vhs doc/demo.tape
# Requires: vecgrep on PATH, vhs installed (brew install vhs)

Output doc/demo.gif
Set Width 1200
Set Height 700
Set FontSize 16
Set Shell zsh
Set Theme "Catppuccin Mocha"

# Pre-clear cache so indexing is visible
Hide
Type "vecgrep --clear-cache ./src > /dev/null 2>&1"
Enter
Sleep 1s
Show

# Beat 1: Cold start — TUI opens, indexing runs live
Type "vecgrep -i --open-cmd 'nvim +{line} {file}' ./src"
Sleep 500ms
Enter
Sleep 4s

# Beat 2: Type first query — results appear
Type "error handling"
Sleep 2500ms

# Beat 3: Navigate results — preview updates
Down
Sleep 800ms
Down
Sleep 800ms
Down
Sleep 1200ms

# Beat 4: Change query — completely different results
Backspace 14
Sleep 300ms
Type "embedding"
Sleep 2500ms

# Beat 5: Navigate new results
Down
Sleep 800ms
Down
Sleep 1200ms

# Beat 6: Toggle preview off and on
Tab
Sleep 1200ms
Tab
Sleep 1200ms

# Beat 7: Exit
Escape
Sleep 500ms
