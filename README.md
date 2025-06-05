The aim is to solve MPC problems using the GNN approach.
python3 train.py --type sol --epochs 100 --gpu 0
python3 test.py   --test_data_folder ./test_data   --model_folder ./saved_models   --model_type sol   --num_test_samples 2000   --emb_size 32   --gpu 0   --save_results   --verbose
