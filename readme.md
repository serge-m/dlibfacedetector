

# Building dlib
Prepare cmake and gcc:
```
sudo apt install cmake g++ libx11-dev
```


Install boost-python in your python environment. I use conda, thus:
```commandline
conda install boost
```


Get a snapshot of dlib:
```
git clone git@github.com:davisking/dlib.git
cd dlib/
git checkout v19.4 
```

build as written in dlib [readme](https://github.com/davisking/dlib/tree/v19.4):
```
mkdir build; cd build; cmake .. ; cmake --build .
```
(for faster build use `mkdir build; cd build; cmake .. ; cmake --build . -- -j 8` instead)


build python bindings:
```commandline
cd ..
python setup.py install
```
