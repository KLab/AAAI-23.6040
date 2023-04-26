# GenéLive! Generating Rhythm Actions in Love Live!

This repository provides the source code and trained model to reproduce the results in the following paper:

Atsushi Takada, Daichi Yamazaki, Likun Liu, Yudai Yoshida, Nyamkhuu Ganbat, Takayuki Shimotomai, Taiga Yamamoto, Daisuke Sakurai, Naoki Hamada, "GenéLive! Generating Rhythm Actions in Love Live!", [DOI: 10.48550/arXiv.2202.12823](https://doi.org/10.48550/arXiv.2202.12823), to appear at [AAAI-23](https://aaai.org/Conferences/AAAI-23).


## Implementations of the Proposed Method

The proposed method contains two novel techniques:
* _Beat guide_ is implemented in `notes_generator/models/beats.py` file.
* _Multi-scale conv-stack_ is implemented as `ConvStackV7` class in `notes_generator/layers/base_layers.py` file.


## Getting Started

Using this repository, you can reproduce the results based on the Stepmania dataset in the paper. (The results on the other datasets are excluded for copyrights reasons.)
The proposed _GenéLive!_ model trained on the Stepmain dataset is also provided to generate a chart for your own audio file. The chart is a MIDI file that you can open in DAW software and play it.


### Prerequisites


#### Packages

* libsndfile
  * Linux (Debian)
    ```shell
    apt-get install libsndfile1
    ```
  * MacOS
    ```sh
    brew install libsndfile
    ```


#### Python environment

To run the code, a conda environment is required.
We recommend Miniconda, see [Conda official site](https://docs.conda.io/en/latest/miniconda.html) for how to install it.


### Installation

1. Create and activate a new conda environment
   ```sh
   conda create -n my-mlflow
   conda activate my-mlflow
   ```
2. Install Mlflow to the created environment
   ```sh
   conda install mlflow
   ```
3. Clone the repo
   ```sh
   git clone https://github.com/KLab/AAAI-23.6040.git
   ```


## Usage

All the functionalities in the code can be invoked through `mlflow` command.


### Fetch dataset

Run the following command.
```shell
mlflow run -e fetch --experiment-name fetch .
```

After running the command, `data/` directory is created and data is downloaded under the directory.  
The directory structure is like below:
```
data
└── raw
    ├── fraxtil
    │   ├── Fraxtil's Arrow Arrangements
    │   ├── Fraxtil's Beast Beats
    │   └── Tsunamix III
    └── itg
        ├── In The Groove
        └── In The Groove 2
```


#### Parameters

|   Name       |   Default   |   Description                                           |
|--------------|-------------|---------------------------------------------------------|
|   data_dir   |   data      |   Path to a directory in which raw data is downloaded.  |


### Generate notes

Run the following command.

```shell
mlflow run -e generate --experiment-name generate . \
  -P audio_path=path/to/audio \  # specify a path to the audio file you want to generate notes for
  -P midi_save_path=save/path \
  -P bpm_info=""  # example: [(180.0,600,4)]
```


#### Parameters

|   Name            |   Default                     |   Description                                                                                                                                                                  |
|-------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   model_path      |   pretrained_model/model.pth  |   Path to a model used for prediction. Default is a path to a model which achieved the highest f1_micro score with STEPMANIA_F dataset.                                        |
|   audio_path      |   -                           |   Path to an audio file to input to the model.                                                                                                                                 |
|   midi_save_path  |   data/midi_out               |   Directory to which output MIDI file is saved.                                                                                                                                |
|   bpm_info        |   -                           |   A string that contains BPM information of the song.  The content of a string is python literal of list which contains tuples of (bpm, millisecond from head of song, beat).  |


### Check results

The results are stored in `mlruns/` directory.
You can check them on the browser by the following command:

```shell
mlflow ui
```


### Preprocessing

Before moving forward to **Evaluation** and **Training** section, data preprocessing is required in advance.  
Run the following command.

```shell
mlflow run -e preprocess --experiment-name preprocess .
```


#### Parameters

|   Name      |   Default  |   Description                                      |
|-------------|------------|----------------------------------------------------|
|   data_dir  |   data     |   Path to data directory which contains raw data.  |


### Evaluation

1. Create a directory named `model_test_result` under `data/`.
1. Run the following command.

    ```shell
    mlflow run -e test --experiment-name test .
    ```
   
#### Parameters

|   Name             |   Default                         |   Description                                                                                                                                                                                                    |
|--------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   model_dir        |   data/onsets_model               |   The directory where models are saved.                                                                                                                                                                          |
|   app_name         |   STEPMANIA                       |   The dataset name. Choices: STEPMANIA, STEPMANIA_F, STEPMANIA_I.  If STEPMANIA_F is chosen, only use the songs belonging to “Fraxtil” dataset, and if STEPMANIA_I is chosen, only use “In The Groove” dataset.  |
|   score_dir        |   data/train_data/score_onsets_1  |   The directory containing training labels.                                                                                                                                                                      |
|   mel_dir          |   data/train_data/mel_log         |   The directory containing Mel-spectrograms.                                                                                                                                                                     |
|   seq_length       |   20480                           |   The desired length of the sequence of the input to the model.                                                                                                                                                  |
|   batch            |   1                               |   The mini batch size.                                                                                                                                                                                           |
|   num_layers       |   2                               |   The number of LSTM layers.                                                                                                                                                                                     |
|   onset_weight     |   64                              |   The weight multiplied to positive labels when calculating binary cross entropy loss.                                                                                                                           |
|   with_beats       |   1                               |   If set to 1, model accept beat information.                                                                                                                                                                    |
|   conv_stack_type  |   v7                              |   The type of convolution stack.  Choices: v1 (DDC), v7 (GenéLive!).                                                                                                                                             |
|   csv_save_dir     |   data/model_test_result          |   The directory the evaluation result will be saved.                                                                                                                                                             |


#### Note

Since it requires 179 GB of models to fully reproduce the results in the paper, we couldn't include all models in this repository.
Instead, we provide commands to reproduce equivalent experiments in next section.


### Training

1. Create a directory named `onsets_model` under `data/`.
1. Run the following command.

    ```shell
    mlflow run -e train --experiment-name train .
    ```
   

#### Parameters

|   Name                   |   Default                         |   Description                                                                                                                                                                                                    |
|--------------------------|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   app_name               |   STEPMANIA                       |   The dataset name. Choices: STEPMANIA, STEPMANIA_F, STEPMANIA_I.  If STEPMANIA_F is chosen, only use the songs belonging to “Fraxtil” dataset, and if STEPMANIA_I is chosen, only use “In The Groove” dataset. |
|   model_dir              |   data/onset_model                |   The directory to save models.                                                                                                                                                                                      |
|   score_dir              |   data/train_data/score_onsets_1  |   The directory containing training labels.                                                                                                                                                                          |
|   mel_dir                |   data/train_data/mel_log         |   The directory containing Mel-spectrograms.                                                                                                                                                                        |
|   resume                 |   0                               |   If nonzero value is specified, resume training from specified step.                                                                                                                                                |
|   epochs                 |   35                              |   The epoch of training.                                                                                                                                                                                            |
|   batch                  |   32                              |   The mini batch size.                                                                                                                                                                                              |
|   lr_start               |   9e-4                            |   The start value of learning rate scheduling.                                                                                                                                                                      |
|   lr_end                 |   9e-4                            |   The end rate of learning rate scheduling                                                                                                                                                                          |
|   lr_scheduler           |   CosineAnnealingLR               |   The type of learning rate scheduler.  Choices: CosineAnnealingLR, CyclicLR,                                                                                                                                        |
|   seq_length             |   20480                           |   The desired length of the sequence of the input to the model.                                                                                                                                                      |
|   aug_count              |   0                               |   The augmentation count of mel-spectrogram.                                                                                                                                                                        |
|   num_layers             |   2                               |   The number of LSTM layers.                                                                                                                                                                                        |
|   onset_weight           |   64                              |   The weight multiplied to positive labels when calculating binary cross entropy loss.                                                                                                                              |
|   dropout                |   0.3                             |   The dropout rate.                                                                                                                                                                                                  |
|   fuzzy_width            |   5                               |   The width of fuzzy label.                                                                                                                                                                                          |
|   fuzzy_scale            |   0.2                             |   The scale of fuzzy label.                                                                                                                                                                                          |
|   with_beats             |   1                               |   Either 0 or 1. If set to 1, train a model with beat information.                                                                                                                                                  |
|   difficulties           |   -                               |   If a tuple of difficulty ids is set, train a model with data of solely specified difficulties.                                                                                                                    |
|   send_model             |   0                               |   If set to 1, save the best model in the local millruns directory.                                                                                                                                                  |
|   n_saved_model          |   20                              |   The maximum number of recent models saved.                                                                                                                                                                        |
|   log_artifacts          |   ""                              |   The comma separated string which contains path to files wanted to be saved in the local mlruns directory.                                                                                                          |
|   augmentation_setting   |   loader_aug_config.yaml          |   The yaml file containing audio augmentation settings.                                                                                                                                                              |
|   warmup_steps           |   400                             |   The warming up steps for learning rate scheduler.                                                                                                                                                                  |
|   weight_decay           |   0                               |   The L2 regularization coefficient.                                                                                                                                                                                |
|   is_parallel            |   0                               |   If set to 1, use multiple GPU if available.                                                                                                                                                                        |
|   eta_min                |   1e-6                            |   The hyper-parameter for CosineAnnealingLR scheduler.                                                                                                                                                              |
|   conv_stack_type        |   v7                              |   The type of convolution stack.  Choices: v1 (DDC), v7 (GenéLive!).                                                                                                                                                |
|   pretrained_model_path  |   ""                              |   If specified, load weights before start training.                                                                                                                                                                  |
|   rnn_dropout            |   0.1                             |   The dropout rate in RNN layers.                                                                                                                                                                                    |


#### Commands for Reproducing Experiments

|   Experiment                               |   Dataset        |   Command                                                                                          |
|--------------------------------------------|------------------|----------------------------------------------------------------------------------------------------|
|   Proposed method                          |   Fraxtil        |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_F`                          |
|   Beat guide ablation                      |   Fraxtil        |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_F -P with_beats=0`          |
|   Difficulty ablation (Beginner only)      |   Fraxtil        |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_F -P difficulties='(10,)'`  |
|   Difficulty ablation (Intermediate only)  |   Fraxtil        |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_F -P difficulties='(20,)'`  |
|   Difficulty ablation (Advanced only)      |   Fraxtil        |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_F -P difficulties='(30,)'`  |
|   Difficulty ablation (Expert only)        |   Fraxtil        |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_F -P difficulties='(40,)'`  |
|   Conv-stack ablation                      |   Fraxtil        |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_F -P conv_stack_type=v1`    |
|   Proposed method                          |   In the Groove  |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_I`                          |
|   Beat guide ablation                      |   In the Groove  |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_I -P with_beats=0`          |
|   Difficulty ablation (Beginner only)      |   In the Groove  |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_I -P difficulties='(10,)'`  |
|   Difficulty ablation (Intermediate only)  |   In the Groove  |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_I -P difficulties='(20,)'`  |
|   Difficulty ablation (Advanced only)      |   In the Groove  |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_I -P difficulties='(30,)'`  |
|   Difficulty ablation (Expert only)        |   In the Groove  |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_I -P difficulties='(40,)'`  |
|   Conv-stack ablation                      |   In the Groove  |   `mlflow run -e train --experiment-name train . -P app_name=STEPMANIA_I -P conv_stack_type=v1`    |


### Reproducing figures in the paper

See `figure.ipynb` file.


## License

- The code in `notes_generator/ddc/` directory is re-distributed under the MIT License by Chris Donahue. See `notes_generator/ddc/LICENSE` file and [the original repository](https://github.com/chrisdonahue/ddc) for more information.
- The other code is distributed under the MIT License by KLab Inc. See `LICENSE` file for more information.
