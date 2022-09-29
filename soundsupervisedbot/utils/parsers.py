import argparse
from soundsupervisedbot.tasks.rattle import RattleTask
from soundsupervisedbot.tasks.vertical_probing import VerticalProbingTask
from soundsupervisedbot.tasks.horizontal_probing import HorizontalProbingTask
from soundsupervisedbot.tasks.fly_swatter import FlySwatterTask
from soundsupervisedbot.tasks.tambourine import TambourineTask



def get_collect_data_parser():
    parser = argparse.ArgumentParser(description='Collect data')

    parser.add_argument('--task-name', required=True,
                        type=str,
                        help='Task name')

    parser.add_argument('--record', default=True,
                        action="store_false",
                        help='record trajectory : if set, does not save details')

    parser.add_argument('--n', default=1,
                        type=int,
                        help='Number of action samples')

    parser.add_argument('--save-dir', default = '/media/data/soundbot/')

    parser.add_argument('--repeats', default=5,
                        type=int,
                        help='Number of times each action is repeated.')
    #Audio params
    parser.add_argument('--n-mics', default=3,
                        type=int,
                        help='Number of microphones used./ Number of channels')
    parser.add_argument('--sample-rate', default=44100,
                        type=int,
                        help='Audio sample rate')
    return parser

def get_eval_parser():
    parser = argparse.ArgumentParser(description='Eval parser')

    parser.add_argument('--task-name', required=True,
                        type=str,
                        help='Task name')

    parser.add_argument('--model-dir', required=True, type=str)
    parser.add_argument('--model-desc', required=True, 
                        type=str, 
                        help="description of model evaluated. Used to save in folders.")
    parser.add_argument('--record', default=True,
                        action="store_false",
                        help='record trajectory : if set, does not save details')
    
    parser.add_argument('--record-force', default=False,
                        action="store_true",
                        help = "Record force")
    
    parser.add_argument('--save-dir', default = '/media/data/soundbot-evals')

    return parser

def init_task(task_name, **kwargs):
    if task_name == 'vertical_probing':
        return VerticalProbingTask(**kwargs)
    elif task_name == 'horizontal_probing':
        return HorizontalProbingTask(**kwargs)
    elif task_name == 'fly_swatter':
        return FlySwatterTask(**kwargs)
    elif task_name == 'rattle':
        return RattleTask(**kwargs)
    elif task_name == 'tambourine':
        return TambourineTask(**kwargs)
    else:
        raise NotImplementedError