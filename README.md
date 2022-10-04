# Sound supervised robot

### Setup

To setup the package:<br>

Follow instructions to setup [aurl](https://github.com/abitha-thankaraj/audio-robot-learning)<br>

For additional robot dependencies:

1. Install additional audio dependencies with:
    ```
           sudo apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev

    ```

2. Install the required python packages:

    ```
        conda env create -f setup/environment.yaml
        conda activate soundbot
    ```   
3. Install the soundbot library
    ```
    pip install -e .
    ```

### Instructions

For data collection:<br>
    ```
        python collect_data.py --task-name=<task_name>
    ```

For model evaluation:<br>
    ```
        python evaluate_model.py --task-name=<task_name> --model-dir=<model_dir> --model-desc=<model_desc>
    ```
