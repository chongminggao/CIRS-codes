## A guide to reproduce the main results of the paper.

There are three steps to reproduce the results.
1. Run the scripts to run the experiments to produce the log files.
2. Collect the results by copying the log files and pasting them to certain directories.
3. Visualize the results and reproduce the main figures and tables in the paper.

We have finished the first two steps and prepared all logs as required. You can visualize the results by only running the third step. 

### Step 1. Re-run the experiments.

All parameters are all set for reproducing the results. We fix the random seed so that the results will not change each time on the same machine. However, the results can still vary when running on different machines.

Different methods are running in a parallel way by adding `&` at the end of each shell command. Please revise the parameter `--cuda` for each line and distribute them according to the GPU memories of your server. A server with 8 GPUs is recommended. 

Make sure you have installed the required environment and downloaded the datasets (see [here](https://github.com/chongminggao/CIRS-codes#installation)).

The following scripts are used for reproducing Figure 5 in the paper.

1. ##### Train the user model on two datasets. 

```shell
# Make sure you are at the root of the project!

# KuaiEnv, Length == 100 and 30
python CIRS-UserModel-kuaishou.py --cuda 0 --leave_threshold 0 --num_leave_compute 1 --tau 1000 --is_ab --epoch 20 --message "Pair11" & # User model for CIRS.
python CIRS-UserModel-kuaishou.py --cuda 1 --leave_threshold 0 --num_leave_compute 1 --tau    0 --no_ab --epoch 20 --message "Pair1" & # User model for CIRS w/o CI.


# VirtualTaobao, Length == 50
python CIRS-UserModel-taobao.py    --cuda 2 --epoch 10 --max_turn 50 --tau 0.01 --leave_threshold 3 --num_leave_compute 5 --message "taobao tau 0.01" & 
python CIRS-UserModel-taobao.py    --cuda 3 --epoch 10 --max_turn 50 --tau 0 --leave_threshold 3 --num_leave_compute 5 --message "taobao tau 0" & 

# VirtualTaobao, Length == 10
python CIRS-UserModel-taobao.py    --cuda 4 --epoch 10 --tau 1 --leave_threshold 1 --num_leave_compute 5 --message "taobao tau 1" & 
```

The logs and saved models are saved in `/saved_models`. Please wait until all processes terminate before learning the RL policies.

2. ##### Learn the RL policies. 

The baseline models (PD, IPS, DICE, DeepFM, $$\epsilon$$-greedy, UCB, Random, MLP) are also run in this lump.

```shell
## KuaiEnv, Length == 30
python "CIRS-RL-kuaishou.py"   --cuda 1 --tau 10  --seed 0  --gamma_exposure 10  --leave_threshold 0  --num_leave_compute 1 --max_turn 30 --is_ab --epoch 1000 --top_rate 0.8 --read_message "Pair11" --message "K_CIRS_len30" &
python "CIRS-RL-kuaishou.py"   --cuda 0 --tau 0   --seed 0  --leave_threshold 0  --num_leave_compute 1 --max_turn 30 --no_ab --epoch 1000 --top_rate 0.8 --read_message "Pair1" --message "K_CIRSwoCI_len30" &

## KuaiEnv, Length == 100
python "CIRS-RL-kuaishou.py"   --cuda 0 --tau 100 --seed 0 --gamma_exposure 10  --leave_threshold 0  --num_leave_compute 1 --max_turn 100 --is_ab --epoch 200 --top_rate 0.8 --read_message "Pair11"  --message "K_CIRS_len100" &
python "CIRS-RL-kuaishou.py"   --cuda 1 --tau 0   --seed 0   --leave_threshold 0  --num_leave_compute 1 --max_turn 100 --no_ab --epoch 200 --top_rate 0.8 --read_message "Pair1" --message "K_CIRSwoCI_len100" &

python PD-pairwise.py             --cuda 0 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --epoch 200 --message "PD"    &
python DeepFM-IPS-pairwise.py     --cuda 1 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --epoch 200 --message "IPS"   &
python DICE.py                    --cuda 2 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --epoch 200 --message "DICE"  &
python CIRS-UserModel-kuaishou.py --cuda 3 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --tau 0 --no_ab --epoch 200 --is_softmax --message "DeepFM+Softmax" &
python CIRS-UserModel-kuaishou.py --cuda 4 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --tau 0 --no_ab --epoch 200  --epsilon 0.1 --not_softmax --message "K_epsilon-greedy" &
python CIRS-UserModel-kuaishou.py --cuda 5 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --tau 0 --no_ab --epoch 200  --epsilon 1.0 --not_softmax --message "K_Random" &
python CIRS-UserModel-kuaishou.py --cuda 6 --leave_threshold 0 --num_leave_compute 1 --top_rate 0.8 --tau 0 --no_ab --epoch 200 --not_softmax --is_ucb --message "UCB" &

## Taobao, Length == 50
python CIRS-RL-taobao.py --cuda 0  --max_turn 50 --epoch 100 --tau 10 --leave_threshold 3 --num_leave_compute 5 --read_message "taobao tau 0.01" --message "T_CIRS_len50" &
python CIRS-RL-taobao.py --cuda 0  --max_turn 50 --epoch 100 --tau 0  --leave_threshold 3 --num_leave_compute 5 --read_message "taobao tau 0" --message "T_CIRSwoCI_len50" &

## Taobao, Length == 10
python CIRS-RL-taobao.py --cuda 0  --max_turn 10 --epoch 200 --tau 0.1 --leave_threshold 1 --num_leave_compute 5 --read_message "taobao tau 1" --message "T_CIRS_len10" &
python CIRS-RL-taobao.py --cuda 0  --max_turn 10 --epoch 200 --tau 0 --leave_threshold 1 --num_leave_compute 5 --read_message "taobao tau 0" --message "T_CIRSwoCI_len10" &
```

**Note:** Learning on VirtualTaobao is slow and the GPU utilization is very low (near 0). The bottleneck is the VirtualTaobao environment, which computes the rewards on the fly. By contrast, experiments running on KuaiEnv are very efficient.

Wait until all processes terminate. You can inspect the performance in the logs under `/saved_models`.

---

### Step 2. Collect all results by copying the log files.

All log files for visualizing Figure 5 and Table 2 should be saved as shown in the following structure.

We have prepared all files for you, so you can directly conduct step 3 without conducting steps 1 and 2. If you want to run it yourself, you should copy your logs from `/saved_models` and replace the logs of the corresponding methods.

```
CIRS-codes
│   ├── results_all_methods
│   │   ├── kuaishou_len100
│   │   │   ├── [DICE]_2022_12_24-16_43_59.log
│   │   │   ├── [DeepFM+Softmax]_2022_12_24-16_43_59.log
│   │   │   ├── [IPS]_2022_12_25-13_34_03.log
│   │   │   ├── [K_CIRS_len100_r08]_2022_12_24-16_43_58.log
│   │   │   ├── [K_CIRSwoCI_len100_r08]_2022_12_24-08_35_42.log
│   │   │   ├── [K_Random]_2022_12_24-16_43_59.log
│   │   │   ├── [K_epsilon-greedy]_2022_12_24-16_43_59.log
│   │   │   ├── [PD]_2022_12_24-16_43_59.log
│   │   │   └── [UCB]_2022_12_24-16_43_59.log
│   │   ├── kuaishou_len30
│   │   │   ├── [K_CIRS_len30_r08]_2022_12_24-08_35_42.log
│   │   │   └── [K_CIRSwoCI_len30_r08]_2022_12_24-16_43_58.log
│   │   ├── taobao_len10
│   │   │   ├── [T_CIRS_len10]_2022_12_18-00_07_19.log
│   │   │   └── [T_CIRSwoCI_len10]_2022_12_18-00_07_19.log
│   │   └── taobao_len50
│   │       ├── [T_CIRS_len50]_2022_12_17-07_01_33.log
│   │       ├── [T_CIRSwoCI_len50]_2022_12_17-07_01_33.log
│   │       ├── [T_MLP]_2022_12_16-15_14_14.log
│   │       ├── [T_Random]_2022_12_16-15_26_18.log
│   │       └── [T_epsilon-greedy]_2022_12_16-15_14_14.log
│   ├── results_alpha_beta
│   │   ├── DeepFM_Pair11.pt
│   │   ├── DeepFM_params_Pair11.pickle
│   │   └── normed_mat-Pair11.pickle
```

---

### Step 3. Visualize the results.

We have already prepared those logs in this repository. So you can directly visualize the results.

We illustrate the results via the Jupyter Notebook file: [visualize_main_results.ipynb](./visualize_main_results.ipynb).

All generated figures and tables are saved under [figures](figures).

##### - Main results:

![main figure](figures/main_result.pdf)

<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>ways</th>
      <th colspan="4" halign="left">Standard</th>
      <th colspan="4" halign="left">No Overlapping</th>
    </tr>
    <tr>
      <th>metrics</th>
      <th><span class="MathJax_Preview" style="color: inherit;"></span><span id="MathJax-Element-1-Frame" class="mjx-chtml MathJax_CHTML" tabindex="0" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mtext>CV</mtext><mtext>M</mtext></msub></math>" role="presentation" style="font-size: 119%; position: relative;"><span id="MJXc-Node-1" class="mjx-math" aria-hidden="true"><span id="MJXc-Node-2" class="mjx-mrow"><span id="MJXc-Node-3" class="mjx-msubsup"><span class="mjx-base"><span id="MJXc-Node-4" class="mjx-mtext"><span class="mjx-char MJXc-TeX-main-R" style="padding-top: 0.424em; padding-bottom: 0.354em;">CV</span></span></span><span class="mjx-sub" style="font-size: 70.7%; vertical-align: -0.212em; padding-right: 0.071em;"><span id="MJXc-Node-5" class="mjx-mtext" style=""><span class="mjx-char MJXc-TeX-main-R" style="padding-top: 0.424em; padding-bottom: 0.354em;">M</span></span></span></span></span></span><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mtext>CV</mtext><mtext>M</mtext></msub></math></span></span><script type="math/tex" id="MathJax-Element-1">\text{CV}_\text{M}</script></th>
      <th>CV</th>
      <th>Length</th>
      <th>MCD</th>
      <th><span class="MathJax_Preview" style="color: inherit;"></span><span id="MathJax-Element-2-Frame" class="mjx-chtml MathJax_CHTML" tabindex="0" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mtext>CV</mtext><mtext>M</mtext></msub></math>" role="presentation" style="font-size: 119%; position: relative;"><span id="MJXc-Node-6" class="mjx-math" aria-hidden="true"><span id="MJXc-Node-7" class="mjx-mrow"><span id="MJXc-Node-8" class="mjx-msubsup"><span class="mjx-base"><span id="MJXc-Node-9" class="mjx-mtext"><span class="mjx-char MJXc-TeX-main-R" style="padding-top: 0.424em; padding-bottom: 0.354em;">CV</span></span></span><span class="mjx-sub" style="font-size: 70.7%; vertical-align: -0.212em; padding-right: 0.071em;"><span id="MJXc-Node-10" class="mjx-mtext" style=""><span class="mjx-char MJXc-TeX-main-R" style="padding-top: 0.424em; padding-bottom: 0.354em;">M</span></span></span></span></span></span><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mtext>CV</mtext><mtext>M</mtext></msub></math></span></span><script type="math/tex" id="MathJax-Element-2">\text{CV}_\text{M}</script></th>
      <th>CV</th>
      <th>Length</th>
      <th>MCD</th>
    </tr>
    <tr>
      <th>Exp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th><span class="MathJax_Preview" style="color: inherit;"></span><span id="MathJax-Element-3-Frame" class="mjx-chtml MathJax_CHTML" tabindex="0" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><mi>&amp;#x03F5;</mi></math>" role="presentation" style="font-size: 119%; position: relative;"><span id="MJXc-Node-11" class="mjx-math" aria-hidden="true"><span id="MJXc-Node-12" class="mjx-mrow"><span id="MJXc-Node-13" class="mjx-mi"><span class="mjx-char MJXc-TeX-math-I" style="padding-top: 0.214em; padding-bottom: 0.284em;">ϵ</span></span></span></span><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ϵ</mi></math></span></span><script type="math/tex" id="MathJax-Element-3">\epsilon</script>-greedy</th>
      <td>0.102+0.013</td>
      <td>0.014+0.002</td>
      <td>2.319+0.053</td>
      <td>0.981+0.006</td>
      <td>0.101+0.004</td>
      <td>0.108+0.005</td>
      <td>17.758+0.386</td>
      <td>0.782+0.002</td>
    </tr>
    <tr>
      <th>DICE</th>
      <td>0.013+0.001</td>
      <td>0.004+0.000</td>
      <td>5.580+0.301</td>
      <td>0.784+0.013</td>
      <td>0.018+0.001</td>
      <td>0.010+0.001</td>
      <td>9.724+0.514</td>
      <td>0.791+0.005</td>
    </tr>
    <tr>
      <th>DeepFM</th>
      <td>0.013+0.001</td>
      <td>0.004+0.000</td>
      <td>5.603+0.285</td>
      <td>0.783+0.013</td>
      <td>0.018+0.001</td>
      <td>0.010+0.001</td>
      <td>9.690+0.478</td>
      <td>0.790+0.004</td>
    </tr>
    <tr>
      <th>IPS</th>
      <td>0.411+0.113</td>
      <td>0.174+0.044</td>
      <td>7.064+0.585</td>
      <td>0.846+0.018</td>
      <td>0.415+0.111</td>
      <td>0.179+0.042</td>
      <td>7.337+0.763</td>
      <td>0.849+0.016</td>
    </tr>
    <tr>
      <th>PD</th>
      <td>0.012+0.001</td>
      <td>0.003+0.000</td>
      <td>4.833+0.242</td>
      <td>0.805+0.012</td>
      <td>0.014+0.001</td>
      <td>0.009+0.000</td>
      <td>10.842+0.534</td>
      <td>0.794+0.003</td>
    </tr>
    <tr>
      <th>Random</th>
      <td>0.802+0.013</td>
      <td>0.368+0.016</td>
      <td>7.646+0.422</td>
      <td>0.800+0.010</td>
      <td>0.802+0.015</td>
      <td>0.369+0.018</td>
      <td>7.658+0.475</td>
      <td>0.798+0.011</td>
    </tr>
    <tr>
      <th>UCB</th>
      <td>0.003+0.000</td>
      <td>0.000+0.000</td>
      <td>2.082+0.653</td>
      <td>1.000+0.000</td>
      <td>0.005+0.000</td>
      <td>0.005+0.000</td>
      <td>18.000+0.000</td>
      <td>0.778+0.000</td>
    </tr>
    <tr>
      <th>CIRS</th>
      <td>0.085+0.127</td>
      <td>0.068+0.062</td>
      <td>61.508+34.175</td>
      <td>0.392+0.072</td>
      <td>0.109+0.122</td>
      <td>0.100+0.059</td>
      <td>41.091+11.046</td>
      <td>0.401+0.049</td>
    </tr>
    <tr>
      <th>CIRS w/o CI</th>
      <td>0.078+0.137</td>
      <td>0.060+0.075</td>
      <td>70.930+34.681</td>
      <td>0.443+0.090</td>
      <td>0.108+0.127</td>
      <td>0.100+0.064</td>
      <td>40.386+9.175</td>
      <td>0.426+0.042</td>
    </tr>
  </tbody>
</table>
</div>
##### - Results for study on parameters $\alpha$ and $\beta$.

| <img src="figures/alpha_popularity.pdf" alt="main figure" style="zoom: 67%;" /> | <img src="figures/beta_popularity.pdf" alt="main figure" style="zoom:67%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

##### - Results for study on different leave conditions on two datasets.

![main figure](figures/leave.pdf)
