@echo off
pushd "%~dp0"
id -g | sed 's@$@ ../../Config/Keys/id_rsa_nao@' | xargs chgrp
..\..\Util\mare\Windows\bin\mare.exe --vcxproj=2015
popd
