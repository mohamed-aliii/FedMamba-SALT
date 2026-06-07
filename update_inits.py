import os
import re

files_to_update = [
    'train_fed_finetune.py',
    'train_fedavg.py',
    'train_centralized.py',
    'eval/linear_probe.py',
    'diagnostic_teacher_probe.py',
    'tests/test_end_to_end.py',
    'tests/test_student.py'
]

pattern = re.compile(r'InceptionMambaEncoder\s*\(\s*patch_size=[^)]+\)', re.DOTALL)

for fpath in files_to_update:
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r') as f:
        content = f.read()
    
    new_content = pattern.sub('InceptionMambaEncoder()', content)
    
    with open(fpath, 'w') as f:
        f.write(new_content)
    print(f'Updated {fpath}')
