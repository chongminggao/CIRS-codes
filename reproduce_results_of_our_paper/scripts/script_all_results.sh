


# %% UserModel

# Taobao, Length == 50

python CIRS-UserModel-taobao.py    --cuda 0 --epoch 10 --max_turn 50 --tau 0.01 --leave_threshold 3 --num_leave_compute 5 --message "taobao tau 0.01" & # UM Taobao50 CIRS
python CIRS-UserModel-taobao.py    --cuda 0 --epoch 10 --max_turn 50 --tau 0 --leave_threshold 3 --num_leave_compute 5 --message "taobao tau 0" & # UM CIRS-wo-CI

# baselines
python MLP-epsilonGreedy-taobao.py --cuda 0 --epoch 100 --max_turn 50 --epsilon 0.1  --leave_threshold 3 --num_leave_compute 5 --message "T_epsilon-greedy" &
python MLP-epsilonGreedy-taobao.py --cuda 0 --epoch 100 --max_turn 50 --epsilon 1  --leave_threshold 3 --num_leave_compute 5 --message "T_Random" &
python MLP-taobao.py               --cuda 0 --epoch 100 --max_turn 50  --leave_threshold 3 --num_leave_compute 5 --message "T_MLP" &

# Taobao, Length == 10
python CIRS-UserModel-taobao.py    --cuda 0 --epoch 10 --tau 1 --leave_threshold 1 --num_leave_compute 5 --message "taobao tau 1" & # UM CIRS



# KuaiEnv, Length == 100
# baselines
python PD-pairwise.py             --cuda 0 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --epoch 200 --message "PD"    &
python DeepFM-IPS-pairwise.py     --cuda 1 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --epoch 200 --message "IPS"   &
python DICE.py                    --cuda 2 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --epoch 200 --message "DICE"  &
python CIRS-UserModel-kuaishou.py --cuda 3 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --tau 0 --no_ab --epoch 200 --is_softmax --message "DeepFM+Softmax" &
python CIRS-UserModel-kuaishou.py --cuda 4 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --tau 0 --no_ab --epoch 200  --epsilon 0.1 --not_softmax --message "K_epsilon-greedy" &
python CIRS-UserModel-kuaishou.py --cuda 5 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --tau 0 --no_ab --epoch 200  --epsilon 1.0 --not_softmax --message "K_Random" &
python CIRS-UserModel-kuaishou.py --cuda 6 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --tau 0 --no_ab --epoch 200 --not_softmax --is_ucb --message "UCB" &
#
#
## KuaiEnv, Length == 30 User models
python CIRS-UserModel-kuaishou.py --cuda 4 --leave_threshold 0 --num_leave_compute 1 --tau 1000 --is_ab --epoch 20 --message "Pair11" & # UM CIRS
python CIRS-UserModel-kuaishou.py --cuda 4 --leave_threshold 0 --num_leave_compute 1 --tau    0 --no_ab --epoch 20 --message "Pair1" & # UM CIRS-wo-CI


## %% Policy Model
#
# Taobao, Length == 50
python CIRS-RL-taobao.py --cuda 0  --max_turn 50 --epoch 100 --tau 10 --leave_threshold 3 --num_leave_compute 5 --read_message "taobao tau 0.01" --message "T_CIRS_len50" &
python CIRS-RL-taobao.py --cuda 0  --max_turn 50 --epoch 100 --tau 0  --leave_threshold 3 --num_leave_compute 5 --read_message "taobao tau 0" --message "T_CIRSwoCI_len50" &

# Taobao, Length == 10
python CIRS-RL-taobao.py --cuda 0  --max_turn 10 --epoch 200 --tau 0.1 --leave_threshold 1 --num_leave_compute 5 --read_message "taobao tau 1" --message "T_CIRS_len10" &
python CIRS-RL-taobao.py --cuda 0  --max_turn 10 --epoch 200 --tau 0 --leave_threshold 1 --num_leave_compute 5 --read_message "taobao tau 0" --message "T_CIRSwoCI_len10" &
#
#
# KuaiEnv, Length == 30
python "CIRS-RL-kuaishou.py"   --cuda 4 --tau 10  --seed 0  --gamma_exposure 10  --leave_threshold 0  --num_leave_compute 1 --max_turn 30 --is_ab --epoch 1000 --read_message "Pair11" --message "K_CIRS_len30" &
python "CIRS-RL-kuaishou.py"   --cuda 5 --tau 0   --seed 0  --leave_threshold 0  --num_leave_compute 1 --max_turn 30 --no_ab --epoch 1000 --read_message "Pair1" --message "K_CIRSwoCI_len30" &

### KuaiEnv, Length == 100
python "CIRS-RL-kuaishou.py"   --cuda 6 --tau 100 --seed 0 --gamma_exposure 10  --leave_threshold 0  --num_leave_compute 1 --max_turn 100 --is_ab --epoch 200 --read_message "Pair11"  --message "K_CIRS_len100" &
python "CIRS-RL-kuaishou.py"   --cuda 7 --tau 0   --seed 0   --leave_threshold 0  --num_leave_compute 1 --max_turn 100 --no_ab --epoch 200 --read_message "Pair1" --message "K_CIRSwoCI_len100" &

