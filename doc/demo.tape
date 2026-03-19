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

# Beat 1: Clear cache and launch TUI with nvim opener
Type "vecgrep --clear-cache && vecgrep -i 'streaming results'"
Sleep 500ms
Enter

# Wait for model load + indexing
Sleep 6s

# Beat 2: Navigate results
Down
Sleep 1s
Down
Sleep 1s
Down
Sleep 1500ms

# Beat 3: Change query
Backspace 17
Sleep 500ms
Type "embedding"
Sleep 3s

# Beat 4: Navigate new results
Down
Sleep 1s
Down
Sleep 1500ms

# Beat 5: Toggle preview
Tab
Sleep 1500ms
Tab
Sleep 1500ms

# Beat 6: Open in nvim
Enter
Sleep 2s
Up
Sleep 100ms
Down
Sleep 100ms
Down
Sleep 1s

# Beat 7: Exit nvim
Type ":q"
Enter
Sleep 1s

Type "vecgrep -q -l 'using config from files' ./src"
Enter
Sleep 2s
