CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ../O-GEE/transformers/examples/pytorch/question-answering/run_qa.py \
  --model_name_or_path albert-base-v2  \
  --do_train \
  --ddp_timeout 18000 \
  --do_eval \
  --train_file re_data/sparse_re_train2.json \
  --validation_file re_data/sparse_re_dev2.json \
  --test_file re_data/sparse_re_test2.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir models/wde_re_model \
  --save_steps 10000 \
  --version_2_with_negative 

