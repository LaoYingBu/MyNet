@echo off
cd /d E:\caffe-gpu\caffe-windows\Build\x64\Release\pycaffe
python ^
draw_net.py D:\MyNet\huawei2015_incept4_train.prototxt D:\MyNet\incept.png --rankdir=TB
cmd