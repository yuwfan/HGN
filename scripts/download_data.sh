BLOB='https://convaisharables.blob.core.windows.net/hgn'

wget $BLOB/data.tar.gz
tar -xzvf data.tar.gz
rm data.tar.gz
