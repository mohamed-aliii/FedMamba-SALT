import json
import os

with open('d:/_Graduation/fedmamba_salt/notebooks/FedMamba_SALT_Centralized.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
# Find index of Section 9
insert_idx = len(cells)
for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown' and len(c['source']) > 0 and '## Section 9:' in ''.join(c['source']):
        insert_idx = i
        break

new_cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '### 8.3 — Fine-Tuning with 30% Label Scarcity\n', 
            '\n', 
            'Trains the model using only 30% of the available training data to evaluate label robustness.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'EVAL_FT_30_DIR = os.path.join(OUTPUT_DIR, "eval_full_finetune_30pct")\n',
            '\n',
            '!python -m eval.linear_probe \\\n',
            '    --encoder_ckpt "{FINAL_CKPT}" \\\n',
            '    --data_path "{DRIVE_DATASET}" \\\n',
            '    --num_classes {NUM_CLASSES} \\\n',
            '    --output_dir "{EVAL_FT_30_DIR}" \\\n',
            '    --epochs 50 \\\n',
            '    --batch_size 256 \\\n',
            '    --lr 1e-3 \\\n',
            '    --mode full_finetune \\\n',
            '    --label_fraction 0.3\n',
            '\n',
            'curves_30 = os.path.join(EVAL_FT_30_DIR, "finetune_30pct", "training_curves_full_finetune_30pct.png")\n',
            'cm_30 = os.path.join(EVAL_FT_30_DIR, "confusion_matrix_full_finetune_30pct.png")\n',
            'if os.path.exists(curves_30): display(Image(filename=curves_30))\n',
            'if os.path.exists(cm_30): display(Image(filename=cm_30))'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '### 8.4 — Fine-Tuning with 60% Label Scarcity\n', 
            '\n', 
            'Trains the model using 60% of the available training data.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'EVAL_FT_60_DIR = os.path.join(OUTPUT_DIR, "eval_full_finetune_60pct")\n',
            '\n',
            '!python -m eval.linear_probe \\\n',
            '    --encoder_ckpt "{FINAL_CKPT}" \\\n',
            '    --data_path "{DRIVE_DATASET}" \\\n',
            '    --num_classes {NUM_CLASSES} \\\n',
            '    --output_dir "{EVAL_FT_60_DIR}" \\\n',
            '    --epochs 50 \\\n',
            '    --batch_size 256 \\\n',
            '    --lr 1e-3 \\\n',
            '    --mode full_finetune \\\n',
            '    --label_fraction 0.6\n',
            '\n',
            'curves_60 = os.path.join(EVAL_FT_60_DIR, "finetune_60pct", "training_curves_full_finetune_60pct.png")\n',
            'cm_60 = os.path.join(EVAL_FT_60_DIR, "confusion_matrix_full_finetune_60pct.png")\n',
            'if os.path.exists(curves_60): display(Image(filename=curves_60))\n',
            'if os.path.exists(cm_60): display(Image(filename=cm_60))'
        ]
    }
]

nb['cells'][insert_idx:insert_idx] = new_cells

with open('d:/_Graduation/fedmamba_salt/notebooks/FedMamba_SALT_Centralized.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
