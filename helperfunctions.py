import os

def get_next_run_number(base_dir):

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 1

    dirs = os.listdir(base_dir)
    max_run_number = 0
    for d in dirs:
        if d.startswith('real') and d[4:].isdigit():
            run_number = int(d[4:])
            if run_number > max_run_number:
                max_run_number = run_number
                
    return max_run_number + 1

