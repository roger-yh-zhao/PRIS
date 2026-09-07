python pdb2graph.py
python test_model.py -m weight_MS2.pth -featsl 27 -e gps -gps_local ASGAT -gps_global SparseAttention
python pwm2score.py
python metric.py