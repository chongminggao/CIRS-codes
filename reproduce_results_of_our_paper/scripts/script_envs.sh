## For KuaiEnv:

python DICE.py                     --cuda 6 --leave_threshold 0 --num_leave_compute 1 --epoch 10 --message "DICE-leave1"
python DeepFM-IPS-pairwise.py      --cuda 6 --leave_threshold 0 --num_leave_compute 1 --epoch 10 --message "IPS-leave1"
python PD-pairwise.py              --cuda 6 --leave_threshold 0 --num_leave_compute 1 --epoch 10 --message "PD-leave1"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 1 --tau 0 --no_ab --epoch 10 --not_softmax --is_ucb --message "UCB-leave1"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 1 --tau 0 --no_ab --epoch 10 --is_softmax --message "DeepFM-leave1"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 1 --tau 0 --no_ab --epoch 10  --epsilon 0.1 --not_softmax --message "Epsilon-greedy-leave1"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 1 --tau 0 --no_ab --epoch 10  --epsilon 1.0 --not_softmax --message "Random-leave1"

python DICE.py                     --cuda 6 --leave_threshold 0 --num_leave_compute 2 --epoch 10 --message "DICE-leave2"
python DeepFM-IPS-pairwise.py      --cuda 6 --leave_threshold 0 --num_leave_compute 2 --epoch 10 --message "IPS-leave2"
python PD-pairwise.py              --cuda 6 --leave_threshold 0 --num_leave_compute 2 --epoch 10 --message "PD-leave2"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 2 --tau 0 --no_ab --epoch 10 --not_softmax --is_ucb --message "UCB-leave2"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 2 --tau 0 --no_ab --epoch 10 --is_softmax --message "DeepFM-leave2"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 2 --tau 0 --no_ab --epoch 10  --epsilon 0.1 --not_softmax --message "Epsilon-greedy-leave2"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 2 --tau 0 --no_ab --epoch 10  --epsilon 1.0 --not_softmax --message "Random-leave2"

python DICE.py                     --cuda 6 --leave_threshold 0 --num_leave_compute 3 --epoch 10 --message "DICE-leave3"
python DeepFM-IPS-pairwise.py      --cuda 6 --leave_threshold 0 --num_leave_compute 3 --epoch 10 --message "IPS-leave3"
python PD-pairwise.py              --cuda 6 --leave_threshold 0 --num_leave_compute 3 --epoch 10 --message "PD-leave3"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 3 --tau 0 --no_ab --epoch 10 --not_softmax --is_ucb --message "UCB-leave3"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 3 --tau 0 --no_ab --epoch 10 --is_softmax --message "DeepFM-leave3"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 3 --tau 0 --no_ab --epoch 10  --epsilon 0.1 --not_softmax --message "Epsilon-greedy-leave3"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 3 --tau 0 --no_ab --epoch 10  --epsilon 1.0 --not_softmax --message "Random-leave3"

python DICE.py                     --cuda 6 --leave_threshold 0 --num_leave_compute 4 --epoch 10 --message "DICE-leave4"
python DeepFM-IPS-pairwise.py      --cuda 6 --leave_threshold 0 --num_leave_compute 4 --epoch 10 --message "IPS-leave4"
python PD-pairwise.py              --cuda 6 --leave_threshold 0 --num_leave_compute 4 --epoch 10 --message "PD-leave4"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 4 --tau 0 --no_ab --epoch 10 --not_softmax --is_ucb --message "UCB-leave4"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 4 --tau 0 --no_ab --epoch 10 --is_softmax --message "DeepFM-leave4"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 4 --tau 0 --no_ab --epoch 10  --epsilon 0.1 --not_softmax --message "Epsilon-greedy-leave4"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 4 --tau 0 --no_ab --epoch 10  --epsilon 1.0 --not_softmax --message "Random-leave4"

python DICE.py                     --cuda 6 --leave_threshold 0 --num_leave_compute 5 --epoch 10 --message "DICE-leave5"
python DeepFM-IPS-pairwise.py      --cuda 6 --leave_threshold 0 --num_leave_compute 5 --epoch 10 --message "IPS-leave5"
python PD-pairwise.py              --cuda 6 --leave_threshold 0 --num_leave_compute 5 --epoch 10 --message "PD-leave5"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 5 --tau 0 --no_ab --epoch 10 --not_softmax --is_ucb --message "UCB-leave5"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 5 --tau 0 --no_ab --epoch 10 --is_softmax --message "DeepFM-leave5"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 5 --tau 0 --no_ab --epoch 10  --epsilon 0.1 --not_softmax --message "Epsilon-greedy-leave5"
python CIRS-UserModel-kuaishou.py  --cuda 6 --leave_threshold 0 --num_leave_compute 5 --tau 0 --no_ab --epoch 10  --epsilon 1.0 --not_softmax --message "Random-leave5"

#
python CIRS-RL-kuaishou.py  --cuda 0 --seed 0 --leave_threshold 0 --num_leave_compute 1 --tau 10  --gamma_exposure 10  --max_turn 30 --is_ab --epoch 1000 --read_message "Pair11"  --message "CIRS-leave1"  &
python CIRS-RL-kuaishou.py  --cuda 1 --seed 0 --leave_threshold 0 --num_leave_compute 2 --tau 10  --gamma_exposure 10  --max_turn 30 --is_ab --epoch 1000 --read_message "Pair11"  --message "CIRS-leave2"  &
python CIRS-RL-kuaishou.py  --cuda 2 --seed 0 --leave_threshold 0 --num_leave_compute 3 --tau 10  --gamma_exposure 10  --max_turn 30 --is_ab --epoch 1000 --read_message "Pair11"  --message "CIRS-leave3"  &
python CIRS-RL-kuaishou.py  --cuda 3 --seed 0 --leave_threshold 0 --num_leave_compute 4 --tau 10  --gamma_exposure 10  --max_turn 30 --is_ab --epoch 1000 --read_message "Pair11"  --message "CIRS-leave4"  &
python CIRS-RL-kuaishou.py  --cuda 4 --seed 0 --leave_threshold 0 --num_leave_compute 5 --tau 10  --gamma_exposure 10  --max_turn 30 --is_ab --epoch 1000 --read_message "Pair11"  --message "CIRS-leave5"  &

python CIRS-RL-kuaishou.py --cuda 5  --seed 0 --leave_threshold 0 --num_leave_compute 1 --tau 0   --max_turn 30 --no_ab --epoch 1000 --read_message "Pair1" --message "CIRS w_o CI-leave1" &
python CIRS-RL-kuaishou.py --cuda 6  --seed 0 --leave_threshold 0 --num_leave_compute 2 --tau 0   --max_turn 30 --no_ab --epoch 1000 --read_message "Pair1" --message "CIRS w_o CI-leave2" &
python CIRS-RL-kuaishou.py --cuda 7  --seed 0 --leave_threshold 0 --num_leave_compute 3 --tau 0   --max_turn 30 --no_ab --epoch 1000 --read_message "Pair1" --message "CIRS w_o CI-leave3" &
python CIRS-RL-kuaishou.py --cuda 0  --seed 0 --leave_threshold 0 --num_leave_compute 4 --tau 0   --max_turn 30 --no_ab --epoch 1000 --read_message "Pair1" --message "CIRS w_o CI-leave4" &
python CIRS-RL-kuaishou.py --cuda 1  --seed 0 --leave_threshold 0 --num_leave_compute 5 --tau 0   --max_turn 30 --no_ab --epoch 1000 --read_message "Pair1" --message "CIRS w_o CI-leave5" &


#
### For VirtualTaobao:

python "CIRS-RL-taobao.py"  --leave_threshold 5.0  --cuda 0 --epoch 200 --max_turn 10  --tau 0     --read_message "taobao tau 0" --message "CIRS w_o CI-leave5" # lab
python "CIRS-RL-taobao.py"  --leave_threshold 4.0  --cuda 0 --epoch 200 --max_turn 10  --tau 0     --read_message "taobao tau 0" --message "CIRS w_o CI-leave4" # lab
python "CIRS-RL-taobao.py"  --leave_threshold 3.0  --cuda 0 --epoch 200 --max_turn 10  --tau 0     --read_message "taobao tau 0" --message "CIRS w_o CI-leave3" # lab5
python "CIRS-RL-taobao.py"  --leave_threshold 2.0  --cuda 0 --epoch 200 --max_turn 10  --tau 0     --read_message "taobao tau 0" --message "CIRS w_o CI-leave2" # lab5
python "CIRS-RL-taobao.py"  --leave_threshold 1.0  --cuda 0 --epoch 200 --max_turn 10  --tau 0     --read_message "taobao tau 0" --message "CIRS w_o CI-leave1" # lab5, lab10


python "CIRS-RL-taobao.py"  --leave_threshold 5.0  --cuda 0 --epoch 200 --max_turn 10  --tau 0.1  --gamma_exposure 10 --read_message "taobao tau 1" --message "CIRS-leave5" # lab10
python "CIRS-RL-taobao.py"  --leave_threshold 4.0  --cuda 0 --epoch 200 --max_turn 10  --tau 0.1  --gamma_exposure 10 --read_message "taobao tau 1" --message "CIRS-leave4" # lab10
python "CIRS-RL-taobao.py"  --leave_threshold 3.0  --cuda 0 --epoch 200 --max_turn 10  --tau 0.1  --gamma_exposure 10 --read_message "taobao tau 1" --message "CIRS-leave3" # lab5, lab10
python "CIRS-RL-taobao.py"  --leave_threshold 2.0  --cuda 2 --epoch 200 --max_turn 10  --tau 0.1  --gamma_exposure 10 --read_message "taobao tau 1" --message "CIRS-leave2" # lab 12, lab5
python "CIRS-RL-taobao.py"  --leave_threshold 1.0  --cuda 2 --epoch 200 --max_turn 10  --tau 0.1  --gamma_exposure 10 --read_message "taobao tau 1" --message "CIRS-leave1" # lab 12, lab5


python "MLP-taobao.py" --cuda 4 --epoch 10 --tau 0  --leave_threshold 1 --message "MLP-leave1" &
python "MLP-taobao.py" --cuda 5 --epoch 10 --tau 0  --leave_threshold 2 --message "MLP-leave2" &
python "MLP-taobao.py" --cuda 6 --epoch 10 --tau 0  --leave_threshold 3 --message "MLP-leave3" &
python "MLP-taobao.py" --cuda 7 --epoch 10 --tau 0  --leave_threshold 4 --message "MLP-leave4" &
python "MLP-taobao.py" --cuda 4 --epoch 10 --tau 0  --leave_threshold 5 --message "MLP-leave5" &

python "MLP-epsilonGreedy-taobao.py" --cuda 6 --epoch 10 --epsilon 1.0 --leave_threshold 1 --message "Random-leave1"
python "MLP-epsilonGreedy-taobao.py" --cuda 6 --epoch 10 --epsilon 1.0 --leave_threshold 2 --message "Random-leave2"
python "MLP-epsilonGreedy-taobao.py" --cuda 6 --epoch 10 --epsilon 1.0 --leave_threshold 3 --message "Random-leave3"
python "MLP-epsilonGreedy-taobao.py" --cuda 6 --epoch 10 --epsilon 1.0 --leave_threshold 4 --message "Random-leave4"
python "MLP-epsilonGreedy-taobao.py" --cuda 6 --epoch 10 --epsilon 1.0 --leave_threshold 5 --message "Random-leave5"

python "MLP-epsilonGreedy-taobao.py" --cuda 6 --epoch 10 --epsilon 0.1 --leave_threshold 1 --message "Epsilon-greedy leave=1" &
python "MLP-epsilonGreedy-taobao.py" --cuda 7 --epoch 10 --epsilon 0.1 --leave_threshold 2 --message "Epsilon-greedy leave=2" &
python "MLP-epsilonGreedy-taobao.py" --cuda 6 --epoch 10 --epsilon 0.1 --leave_threshold 3 --message "Epsilon-greedy leave=3" &
python "MLP-epsilonGreedy-taobao.py" --cuda 7 --epoch 10 --epsilon 0.1 --leave_threshold 4 --message "Epsilon-greedy leave=4" &
python "MLP-epsilonGreedy-taobao.py" --cuda 0 --epoch 10 --epsilon 0.1 --leave_threshold 5 --message "Epsilon-greedy leave=5" &



