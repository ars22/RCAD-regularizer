python train_maxent.py \
--dataset cifar100 \
--model wideresnet \
--data_augmentation \
--lambd 0.2 \
--run_id LS+MPE,alpha:1.0,lambd:0.2  \
--learning_rate 0.1 \
--label_smoothing_factor 0.6 \
--batch_size 128 \
--attack_iters 1 \
--alpha 1.0 \
--seed 10