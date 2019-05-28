echo off
echo Create .githooks folder if not exists"
if not exist ".githooks" mkdir .githooks

echo write 'exec c:/anaconda3/Scripts/pycodestyle.exe --ignore=E501,W504  ./topfarm/' to pre-commit
echo Please modify path to pycodestyle
echo #!/bin/sh > .githooks/pre-commit
echo exec c:/anaconda3/Scripts/pycodestyle.exe --ignore=E501,W504  ./topfarm/ >> .githooks/pre-commit
git config core.hooksPath .githooks
echo Done
pause