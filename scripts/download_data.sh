BLOB='https://convaisharables.blob.core.windows.net/hgn'

wget $BLOB/data.tar.gz
tar -xzvf $BLOB/data.tar.gz
rm $BLOB/data.tar.gz
