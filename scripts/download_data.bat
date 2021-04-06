@echo off

set BLOB=https://convaisharables.blob.core.windows.net/hgn

curl -O %BLOB%/data.tar.gz
tar -xzvf data.tar.gz
REM DEL /Q data.tar.gz
