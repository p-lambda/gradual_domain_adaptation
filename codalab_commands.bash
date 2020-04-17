
cl run :gradual_st :experiments \
"export PYTHONPATH="."; python experiments/gradual_shift_better.py --experiment_name=windowed_vs_accumulate" \
-n windowed_vs_accumulate --request-queue tag=nlp
