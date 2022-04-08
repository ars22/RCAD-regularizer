python train_maxent.py \
--dataset cifar100 \
--model wideresnet \
--data_augmentation \
--lambd 0. \
--run_id LS  \
--learning_rate 0.1 \
--label_smoothing_factor 0.6 \
--batch_size 128 \
--attack_iters 0 \
--alpha 0.0 \
--seed 10