# This project is inspired by this paper: PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation
with the usage of ShuffleNet in neural networks, and removed mixedup nad time/frequency masking, it uses Bibary Cross Entropy as Loss function, and dual microphone technologies to cacel noise.

getting started
pip install -r requirements.txt 

**Where's the code?**

The  model file is in `src/models/Models.py`, the dataloader code is in `src/dataloaders/`, the training and evaluation code is in `src/run_wz_relay_dual.py`.

The recipes are in `egs/[audioset,fsd50k]`, when you run `run.sh`, it will call `src/run.py`, which will then call `src/dataloaders/audioset_dataset.py` and `/src/traintest.py`, which will then call `/src/models/Models.py`. For FSD50K, we provide the data pre-processing code in `egs/fsd50k/prep_fsd.py`, for AudioSet, you need to prepare it by yourself as you need to download audios from YouTube. `run.sh` contains all hyper-parameters of our experiment.

## FSD50K Recipe  

**Step 1. Download the FSD50K dataset from [the official website](https://zenodo.org/record/4060432).**

**Step 2. Prepare the data.**

run:create_dataset/1.create_fsd50k_format.py

This should create three json files wz_relay\wz_relay_train.json

 **Step 3. Generate weigh.**
run: create_dataset/2.gen_weight_file.py

This will automatically generate a set of new Json datafiles in `egs/fsd50k/datafiles`, e.g., `fsd50k_tr_full_type1_2_mean.json` means the datafile with enhanced label set for both Type-I and Type-II error with label modification threshold of mean. You can use these new datafiles as input of `egs/fsd50k/run.sh`, specifically, you can change `p` (label modification threshold) in `[none, mean, median, 5, 10, 25]`. 

If you skipped this step, please set `p` in `egs/fsd50k/run.sh` as `none`. You will get a slightly worse result.

**Step 4. Run the training and evaluation.**

``` 
cd psla/egs/fsd50k
(slurm user) sbatch run.sh
(local user) ./run.sh
```  

The recipe was tested on 4 GTX TITAN GPUs with 12GB memory. The entire training and evaluation takes about 15 hours. Trimming the `target_length` of `run.sh` from 3000 to 1000, and increasing the batch size and learning rate accordingly can significantly reduce the running time, but leads to just slightly worse result.

**Step 5. Get the results.**

The running log will present the results of 1) single model, 2) weight averaging model (i.e., averaging the weight of last few model checkpoints, this does not increase the model size and computational overhead), and 3) checkpoint ensemble models (i.e., averaging the prediction of each epoch) on both the FSD50K official validation set and evaluation set. These results are also saved in `psla/egs/fsd50k/exp/yourexpname/[best_single_,wa_,ensemble_]result.csv`.

We share our training and evaluation log in `psla/egs/fsd50k/exp/`, you can expect to get a similar result as follows. We also share our entire experiment directory at this [dropbox link](https://www.dropbox.com/s/qfaeyuvtse420dn/demo-efficientnet-2-5e-4-fsd50k-impretrain-True-fm48-tm192-mix0.5-bal-True-b24-lemean-2.zip?dl=1). 

```
---------------evaluate best single model on the validation set---------------
mAP: 0.588115
AUC: 0.960351
---------------evaluate best single model on the evaluation set---------------
mAP: 0.558463
AUC: 0.943927
---------------evaluate weight average model on the validation set---------------
mAP: 0.587779
AUC: 0.960226
---------------evaluate weight averages model on the evaluation set---------------
mAP: 0.561647
AUC: 0.943910
---------------evaluate ensemble model on the validation set---------------
mAP: 0.601013
AUC: 0.970726
---------------evaluate ensemble model on the evaluation set---------------
mAP: 0.572588
AUC: 0.955053
```
