#!/bin/bash
#

python bb_fil.py start /home/ubuntu/data/fl_corr01.out beamformer_weights_corr01.dat 1498.75
sleep 25
python bb_fil.py stop 
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr01.fil

python bb_fil.py start /home/ubuntu/data/fl_corr02.out beamformer_weights_corr02.dat 1487.03125
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr02.fil

python bb_fil.py start /home/ubuntu/data/fl_corr03.out beamformer_weights_corr03.dat 1475.3125
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr03.fil

python bb_fil.py start /home/ubuntu/data/fl_corr04.out beamformer_weights_corr04.dat 1463.59375
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr04.fil

python bb_fil.py start /home/ubuntu/data/fl_corr05.out beamformer_weights_corr05.dat 1451.875
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr05.fil

python bb_fil.py start /home/ubuntu/data/fl_corr06.out beamformer_weights_corr06.dat 1440.15625
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr06.fil

python bb_fil.py start /home/ubuntu/data/fl_corr07.out beamformer_weights_corr07.dat 1428.4375
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr07.fil

python bb_fil.py start /home/ubuntu/data/fl_corr08.out beamformer_weights_corr08.dat 1416.71875
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr08.fil

python bb_fil.py start /home/ubuntu/data/fl_corr09.out beamformer_weights_corr09.dat 1405.0
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr09.fil

python bb_fil.py start /home/ubuntu/data/fl_corr10.out beamformer_weights_corr10.dat 1393.28125
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr10.fil

python bb_fil.py start /home/ubuntu/data/fl_corr11.out beamformer_weights_corr11.dat 1381.5625
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr11.fil

python bb_fil.py start /home/ubuntu/data/fl_corr12.out beamformer_weights_corr12.dat 1369.84375
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr12.fil

python bb_fil.py start /home/ubuntu/data/fl_corr13.out beamformer_weights_corr13.dat 1358.125
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr13.fil

python bb_fil.py start /home/ubuntu/data/fl_corr14.out beamformer_weights_corr14.dat 1346.40625
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr14.fil

python bb_fil.py start /home/ubuntu/data/fl_corr15.out beamformer_weights_corr15.dat 1334.6875
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr15.fil

python bb_fil.py start /home/ubuntu/data/fl_corr16.out beamformer_weights_corr16.dat 1322.96875
sleep 25
python bb_fil.py stop
mv /home/ubuntu/tmp/test.fil /home/ubuntu/tmp/corr16.fil










