#for ((N=1;N<9;N++)); do
N=8
python fit_qnm.py --ell 0.0 --Ntones $N
python fit_qnm.py --ell 0.1 --Ntones $N
python fit_qnm.py --ell 0.2 --Ntones $N
python fit_qnm.py --ell 0.226 --Ntones $N
python fit_qnm.py --ell 0.3 --Ntones $N
#done
