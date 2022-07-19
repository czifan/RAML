three sub-stages (5-shot 16+3 setting):

sub-stage1: training close set module
python -u main.py --output_dir ./output_stage1_16 --gpu_id 0,1

sub-stage2: training meta channel module
python -u main.py --finetune --ckpt ./output_stage1_16/final.pth --output_dir ./output_stage2_16/ --total_itrs 10000 --gpu_id 0,1

sub-stage3: training region-aware metric learning module
python -u main_metric.py --ckpt ./output_stage2_16/final.pth --output_dir ./output_stage3_16/  --novel_dir ./novel/

inference:

16+3 5shots:
python main_metric.py --ckpt ./output_stage3_16/final.pth --test_only --test_mode 16_3  --novel_dir ./novel

16+1 5shots:
python main_metric.py --ckpt ./output_stage3_16/final.pth --test_only --test_mode 16_3  --novel_dir ./novel_1

16+1 5shots:
python main_metric.py --ckpt ./output_stage3_16/final.pth --test_only --test_mode 16_1  --novel_dir ./novel

12+7 5shots:
python main_metric.py --ckpt ./output_stage3_12/final.pth --test_only --test_mode 12  --novel_dir ./novel

