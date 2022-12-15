# Length == 50

python CIRS-UserModel-taobao.py --tau 0.01 --read_message "taobao tau 0.01" & # UM Taobao50 CIRS
python CIRS-UserModel-taobao.py --tau 1 --read_message "taobao tau 1" & # UM

python CIRS-UserModel-taobao.py --tau 0 --read_message "taobao tau 0" & # UM CIRS-wo-CI
python CIRS-UserModel-taobao.py --tau 0.1 --read_message "taobao tau 0.1" &


python MLP-epsilonGreedy-taobao.py --epsilon 0.1 --epoch 100 --leave_threshold 3 --num_leave_compute 5 --message "epsilon-greedy" &
python MLP-epsilonGreedy-taobao.py --epsilon 1 --epoch 100 --leave_threshold 3 --num_leave_compute 5 --message "Random" &
python MLP-taobao.py               --epoch 100 --leave_threshold 3 --num_leave_compute 5 --message "MLP" &