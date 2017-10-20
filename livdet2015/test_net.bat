@echo off
cd /d D:\MyNet\livdet2015
E:\caffe-gpu\caffe-windows\Build\x64\Release\caffe.exe train --solver=solver.prototxt 
pause
cmd