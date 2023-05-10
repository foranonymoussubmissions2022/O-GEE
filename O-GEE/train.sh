CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 transformers/examples/pytorch/question-answering/run_qa.py \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --ddp_timeout 18000 \
  --do_eval \
  --train_file training_data/wde_sparse_re_train3.json \
  --validation_file  training_data/wde_sparse_re_dev3.json \
  --test_file  training_data/wde_sparse_re_test3.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir models/wde_re_model2 \
  --save_steps 5000 \
  --version_2_with_negative \
# sparse_albert_base_minority_classes dbp
