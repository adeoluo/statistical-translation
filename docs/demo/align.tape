# Records docs/demo/align.gif from a real run: `vhs docs/demo/align.tape` from the repo root.
Output docs/demo/align.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m statistical_translation align "the red house" "la maison rouge"` Enter
Sleep 2s
Type `python -m statistical_translation translate house` Enter
Sleep 2.5s
