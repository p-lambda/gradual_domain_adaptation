
cl upload data -d "Cover Type Data"
cl upload dataset_32x32 -d "Portraits dataset 32x32"
cl upload gradual_st -d "Gradual self-training library"
cl upload experiments -d "Experiments"

cl run :data :gradual_st :experiments 'export PYTHONPATH="."; python experiments/gradual_shift_better.py --experiment_name=cov_type_main' -n cov_type_main --request-gpus 1 --request-queue nlp
cl run :dataset_32x32.mat :gradual_st :experiments 'export PYTHONPATH="."; python experiments/gradual_shift_better.py --experiment_name=portraits_main' -n portraits_main --request-memory=8g --request-gpus 1 --request-queue nlp
cl run :gradual_st :experiments 'export PYTHONPATH="."; python experiments/gradual_shift_better.py --experiment_name=rotating_mnist_main' -n rotating_mnist_main --request-network --request-memory=8g --request-gpus 1 --request-queue nlp
cl run :gradual_st :experiments 'export PYTHONPATH="."; python experiments/gradual_shift_better.py --experiment_name=gaussian_main' -n gaussian_main --request-gpus 1 --request-queue nlp

cl run :gradual_st :experiments 'export PYTHONPATH="."; python3 experiments/regularization_helps.py --experiment_name=rot_mnist_vary_n' -n rot_mnist_vary_n --request-network --request-memory=8g --request-gpus 1 --request-queue nlp

cl run :data :gradual_st :experiments 'export PYTHONPATH="."; python3 experiments/regularization_helps.py --experiment_name=cov_reg' -n cov_reg --request-gpus 1 --request-queue nlp
cl run :dataset_32x32.mat :gradual_st :experiments 'export PYTHONPATH="."; python3 experiments/regularization_helps.py --experiment_name=portraits_reg' -n portraits_reg --request-memory=8g --request-gpus 1 --request-queue nlp
cl run :gradual_st :experiments 'export PYTHONPATH="."; python3 experiments/regularization_helps.py --experiment_name=rot_mnist_reg' -n rot_mnist_reg --request-network --request-memory=8g --request-gpus 1 --request-queue nlp
cl run :gradual_st :experiments 'export PYTHONPATH="."; python3 experiments/regularization_helps.py --experiment_name=gaussian_reg' -n gaussian_reg --request-gpus 1 --request-queue nlp

cl run :data :gradual_st :experiments 'export PYTHONPATH="."; python3 experiments/regularization_helps.py --experiment_name=cov_soft' -n cov_soft --request-gpus 1 --request-queue nlp
cl run :dataset_32x32.mat :gradual_st :experiments 'export PYTHONPATH="."; python3 experiments/regularization_helps.py --experiment_name=portraits_soft' -n portraits_soft --request-memory=8g --request-gpus 1 --request-queue nlp
cl run :gradual_st :experiments 'export PYTHONPATH="."; python3 experiments/regularization_helps.py --experiment_name=rot_mnist_soft' -n rot_mnist_soft --request-network --request-memory=8g --request-gpus 1 --request-queue nlp
cl run :gradual_st :experiments 'export PYTHONPATH="."; python3 experiments/regularization_helps.py --experiment_name=gaussian_soft' -n gaussian_soft --request-gpus 1 --request-queue nlp

cl run :gradual_st :experiments 'export PYTHONPATH="."; python experiments/gradual_shift_better.py --experiment_name=dialing_ratios_mnist' -n dialing_ratios_mnist --request-network --request-memory=8g --request-gpus 1 --request-queue nlp
