# 3DXMem


## Training
- `python -m torch.distributed.run --master_port 25763 --nproc_per_node=2 train.py --exp_id retrain --stage 4 --num_workers 16 --loader_shuffle False --s4_batch_size 6`

## Evaluation

- `python eval.py --output ../output/astro --mem_every 3 --dataset G --generic_path astro_test_root --save_scores --size -1 --model ./saves/<>.pth `
    - not the checkpoint one