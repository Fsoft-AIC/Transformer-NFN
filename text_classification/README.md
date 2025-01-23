# Text classification with AGNews
Set up datasets:
```bash
mkdir -p data/ag_news && \
    cd data/ag_news && \
    kaggle datasets download -d amananandrai/ag-news-classification-dataset && \
    unzip ag-news-classification-dataset.zip
```

Run

```
python main_text_classification.py --dataset ag_news --batch_size 512 --epochs 100
```

## Sweep
Create a wandb sweep for hyperparameter tuning using the following command:
```
wandb sweep --project "default" --entity <your-entity> sweep_scripts/sweep_config.yml
```

The command will output a `sweep_id` (Ex: `default`). Change the `SWEEP_ID` value in the scripts to this value and run the script to start the sweep:
```
bash sweep_scripts/sweep.sh
```

`sweep.sh` structure:
```bash
NUM_PROCESSS_PER_GPU=8 # number of processes to run on each GPU
GPUS=0,1,2,3,4,5,6,7 # list of GPUs to run the processes on
SWEEP_ID=default # sweep id

# run NUM_PROCESSS_PER_GPU processes on each GPU
for i in $(seq 0 $((NUM_PROCESSS_PER_GPU-1))); do
    for j in $(echo $GPUS | sed "s/,/ /g"); do
        CUDA_VISIBLE_DEVICES=$j wandb agent $SWEEP_ID &
    done
done
```

### Change sweep configuration
The sweep configuration file is located at `sweep_scripts/sweep_config.yml`. You can change the hyperparameters to search for in this file, or create a new one for a different settings.

### Debugging sweep
The command will output a string representing the `sweep_id`. Use this string to start the sweep agent:
```
wandb agent <user-name>/<project-name>/<sweep-id>
```

Start sweep agent for specific GPU:
```
CUDA_VISIBLE_DEVICES=0 wandb agent <user-name>/<project-name>/<sweep-id>
```

We can have **more than one** sweep agent running at the same time. Each agent will pick up a different hyperparameter configuration to train the model.

## Dataset creation
After the sweep is done, we can create a dataset using the best hyperparameters. The dataset creation script is located at `generate.py`. You need to specify the `entity` and `project` of wandb, and the model path to create the dataset. We show an information as below:
```
usage: generate_metadata.py [-h] [--model_dir MODEL_DIR] [--entity ENTITY] [--project_name PROJECT_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
  --entity ENTITY
  --project_name PROJECT_NAME
```

