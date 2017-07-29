

# Installing dlib

dlib requires boost. boost cannot be installed from PyPI

## Installing boost for virtualenv

```
wget -c "http://downloads.sourceforge.net/project/boost/boost/1.57.0/boost_1_57_0.tar.gz"
tar xf boost_1_57_0.tar.gz
cd ./boost_1_57_0
./bootstrap.sh --with-python=$(which python) --prefix=$VIRTUAL_ENV
./b2 --prefix=$VIRTUAL_ENV
./b2 --prefix=$VIRTUAL_ENV install
```

[source](https://gist.github.com/jcsaaddupuy/222675081127b26a92e3)