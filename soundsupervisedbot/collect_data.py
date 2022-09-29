from soundsupervisedbot.tasks.sound_task import record_task
from soundsupervisedbot.utils.parsers import *
from soundsupervisedbot.utils.utils import *
from soundsupervisedbot.utils.slack_me import slack_notification


def main(opts, task):
    primitive = task.sample_primitive()
    for _ in range(opts.repeats):
        record_task(opts.save_dir, task, primitive, 
                record = opts.record, n_mics = opts.n_mics, sample_rate= opts.sample_rate)
    

if __name__ == "__main__":
    
    count = 0
    opts = get_collect_data_parser().parse_args()
    task = init_task(opts.task_name)
    try:
        for i in range(opts.n):
            print("Recording traj {}".format(i))
            main(opts, task)
            count=i

    except Exception as e:
        print(e)
        slack_notification('Data collection stopped: {}'.format(e))
        task.close()

    
    slack_notification('Collected {} data point'.format(count+1))
    task.reset()
    task.close()
