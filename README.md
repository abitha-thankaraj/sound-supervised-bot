# Sound supervised robot
To setup the package:<br>

Follow instructions to setup [aurl](https://github.com/abitha-thankaraj/audio-robot-learning)<br>
For additional robot dependencies:
    
    ```
        sudo apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
        
        conda env create -f setup/environment.yml
        
        conda activate soundbot
        
        pip install -e .
    ```

For data collection:<br>
    ```
        python collect_data.py --task-name=<task_name>
    ```

For model evaluation:<br>
    ```
        python evaluate_model.py --task-name=<task_name> --model-dir=<model_dir> --model-desc=<model_desc>
    ```
