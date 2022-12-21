


# %% UserModel

# Taobao, Length == 50

#python CIRS-UserModel-taobao.py    --cuda 0 --epoch 10 --max_turn 50 --tau 0.01 --leave_threshold 3 --num_leave_compute 5 --message "taobao tau 0.01" & # UM Taobao50 CIRS
#python CIRS-UserModel-taobao.py    --cuda 0 --epoch 10 --max_turn 50 --tau 0 --leave_threshold 3 --num_leave_compute 5 --message "taobao tau 0" & # UM CIRS-wo-CI

#python MLP-epsilonGreedy-taobao.py --cuda 0 --epoch 10 --max_turn 50 --epsilon 0.1 --epoch 100 --leave_threshold 3 --num_leave_compute 5 --message "T_epsilon-greedy" &
#python MLP-epsilonGreedy-taobao.py --cuda 0 --epoch 10 --max_turn 50 --epsilon 1 --epoch 100 --leave_threshold 3 --num_leave_compute 5 --message "T_Random" &
#python MLP-taobao.py               --cuda 0 --epoch 10 --max_turn 50 --epoch 100 --leave_threshold 3 --num_leave_compute 5 --message "T_MLP" &

# Taobao, Length == 10
#python CIRS-UserModel-taobao.py    --cuda 0 --epoch 10 --tau 1 --leave_threshold 1 --num_leave_compute 5 --message "taobao tau 1" & # UM CIRS
# 同上 python CIRS-UserModel-taobao.py    --cuda 0 --epoch 10 --tau 0 --message "taobao tau 0" & # UM CIRS-wo-CI



# KuaiEnv, Length == 50
#python CIRS-UserModel-kuaishou.py --cuda 0 --leave_threshold 0 --num_leave_compute 1 --tau 0 --no_ab --epoch 200 --is_softmax --message "DeepFM+Softmax" &
#python DICE.py                    --cuda 1 --leave_threshold 0 --num_leave_compute 1 --epoch 200 --message "DICE"  &
#python CIRS-UserModel-kuaishou.py --cuda 2 --leave_threshold 0 --num_leave_compute 1 --tau 0 --no_ab --epoch 200  --epsilon 0.9 --not_softmax --message "K_epsilon-greedy" &
#python DeepFM-IPS-pairwise.py     --cuda 1 --leave_threshold 0 --num_leave_compute 1 --epoch 200 --message "IPS"   &
#python PD-pairwise.py             --cuda 3 --leave_threshold 0 --num_leave_compute 1 --epoch 200 --message "PD"    &
#python CIRS-UserModel-kuaishou.py --cuda 0 --leave_threshold 0 --num_leave_compute 1 --tau 0 --no_ab --epoch 200  --epsilon 1.0 --not_softmax --message "K_Random" &
python CIRS-UserModel-kuaishou.py --cuda 1 --leave_threshold 0 --num_leave_compute 1 --tau 0 --no_ab --epoch 200 --not_softmax --is_ucb --message "UCB" &


# KuaiEnv, Length == 30
#python CIRS-UserModel-kuaishou.py --cuda 1 --leave_threshold 0 --num_leave_compute 1 --tau 1000 --is_ab --epoch 10 --message "Pair11" & # UM CIRS
#python CIRS-UserModel-kuaishou.py --cuda 1 --leave_threshold 0 --num_leave_compute 1 --tau    0 --no_ab --epoch 10 --message "Pair1" & # UM CIRS-wo-CI




## %% Policy Model
#
## Taobao, Length == 50
#python CIRS-RL-taobao.py --cuda 0  --max_turn 50 --epoch 100 --tau 10 --leave_threshold 3 --num_leave_compute 5 --read_message "taobao tau 0.01" --message "T_CIRS_len50" &
#python CIRS-RL-taobao.py --cuda 0  --max_turn 50 --epoch 100 --tau 0  --leave_threshold 3 --num_leave_compute 5 --read_message "taobao tau 0" --message "T_CIRSwoCI_len50" &
#
## Taobao, Length == 10
#python CIRS-RL-taobao.py --cuda 0  --max_turn 10 --epoch 200 --tau 0.1 --leave_threshold 1 --num_leave_compute 5 --read_message "taobao tau 1" --message "T_CIRS_len10" &
#python CIRS-RL-taobao.py --cuda 0  --max_turn 10 --epoch 200 --tau 0 --leave_threshold 1 --num_leave_compute 5 --read_message "taobao tau 0" --message "T_CIRSwoCI_len10" &
##
##
## KuaiEnv, Length == 30
#python3.9 "CIRS-RL-kuaishou.py"   --cuda 0 --tau 10    --gamma_exposure 10  --leave_threshold 0  --num_leave_compute 1 --max_turn 30 --is_ab --epoch 1000 --read_message "Pair11" --message "K_CIRS_len30" &
#python3.9 "CIRS-RL-kuaishou.py"   --cuda 1 --tau 0     --leave_threshold 0  --num_leave_compute 1 --max_turn 30 --no_ab --epoch 1000 --read_message "Pair1" --message "K_CIRSwoCI_len30" &
##
##
### KuaiEnv, Length == 100
#python3.9 "CIRS-RL-kuaishou.py"   --cuda 2 --tau 100  --gamma_exposure 10  --leave_threshold 0  --num_leave_compute 1 --max_turn 100 --is_ab --epoch 200 --read_message "Pair11"  --message "K_CIRS_len100" &
#python3.9 "CIRS-RL-kuaishou.py"   --cuda 3 --tau 0      --leave_threshold 0  --num_leave_compute 1 --max_turn 100 --no_ab --epoch 200 --read_message "Pair1" --message "K_CIRSwoCI_len100" & # lds7




