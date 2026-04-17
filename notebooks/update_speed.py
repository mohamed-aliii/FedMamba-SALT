import json
import re

with open('d:/_Graduation/fedmamba_salt/notebooks/FedMamba_SALT_Centralized.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if '!python train_centralized.py' in source or '!python -m eval.linear_probe' in source:
            # Update batch_size
            source = re.sub(r'--batch_size\s+\d+', '--batch_size 512', source)
            # Update num_workers if present
            if '--num_workers' in source:
                source = re.sub(r'--num_workers\s+\d+', '--num_workers 8', source)
            else:
                # Add num_workers
                source = source.replace('--batch_size 512 \\', '--batch_size 512 \\\n    --num_workers 8 \\')
            
            # Split back to lines, preserving newlines
            import io
            lines = []
            for line in source.splitlines(True):
                lines.append(line)
            cell['source'] = lines

with open('d:/_Graduation/fedmamba_salt/notebooks/FedMamba_SALT_Centralized.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
