from soundsupervisedbot.tasks.sound_task import *
from soundsupervisedbot.utils.parsers import *
from soundsupervisedbot.utils.utils import *
from soundsupervisedbot.utils.slack_me import slack_notification
import torchaudio

DATASET_DIR = dict(
    fly_swatter = '/media/data/dataset_fly_swatter/',
    vertical_probing = '/media/data/dataset_vertical_with_momentum',
    horizontal_probing = '/media/data/dataset_horizontal_probing_repeats',
    rattle = '/media/data/dataset_rattle',
    tambourine = '/media/data/dataset_tambourine'
)


def main(opts, task, audio, primitive = None):
    
    if primitive is None: # precomputed primitive for best, random
        primitive = task.predict_primitive(audio)

    print("Predicted: {}".format(primitive))
      
    record_task(eval_save_dir, task, primitive, 
                record = opts.record,
                # Use same cfgs as train recordings
                n_mics = task.evaluator.cfg.dataset.num_mics, 
                sample_rate= task.evaluator.cfg.dataset.sample_rate)

if __name__ == "__main__":

    count = 0
    opts = get_eval_parser().parse_args()
    task = init_task(opts.task_name, model_dir = opts.model_dir)
    
    dataset_parent_dir = DATASET_DIR[opts.task_name]
    
    if 'best' in opts.model_desc:
        fname_act_dict = load_pkl_file(os.path.join('/home/abitha/projects/sound-supervised-bot/cfgs/{}'.format(opts.task_name), 'best-test-act-dict.pkl'))
        fnames = list(fname_act_dict.keys())
        
    elif 'random' in opts.model_desc:
        fname_act_dict = load_pkl_file(os.path.join('/home/abitha/projects/sound-supervised-bot/cfgs/{}'.format(opts.task_name), 'random-test-act-dict.pkl'))
        fnames = list(fname_act_dict.keys())

    else:
        fname_act_dict = None
        fnames = load_pkl_file(os.path.join(opts.model_dir, 'test_fnames.pkl'))
    
    # Uniquify names, actions
    unique_actions = set()
    unique_fnames = set()
    try:
        for i, fname in enumerate(fnames):
            fname = fname.split('/')[-1]
            audio, _ = torchaudio.load(os.path.join(dataset_parent_dir, fname, 'audio-clip.wav'))
            
            original_params = tuple(torch.load(os.path.join(dataset_parent_dir, fname, 'params.pt')).numpy())
            # Run only 200 unique test points on real robot.          
            if original_params in unique_actions:
                continue
            else:
                unique_actions.add(original_params)
                unique_fnames.add(fname)

            print("Original params: {}".format(original_params))
  
            eval_save_dir = os.path.join(opts.save_dir, # Default save loc
                                    opts.task_name, # Task name
                                    opts.model_desc, # Model + notes
                                    os.path.split(opts.model_dir)[-1], # Model identifier - from wandb
                                    fname) #Test point name

            print("recording traj {}".format(i))
            
            if 'random' in opts.model_desc or 'best' in opts.model_desc: # Baselines
                action = fname_act_dict[fname].numpy()
                primitive = task._detorchify(action)

            else:
                primitive = None
                
            main(opts, task, audio, primitive)
            count=i

    except Exception as e:
        print(e)
        slack_notification('Data collection stopped: {}'.format(e))
        task.close()
    
    slack_notification('Collected {} data point'.format(count+1))
    task.reset()
    task.close()