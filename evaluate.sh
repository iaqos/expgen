CUDA_VISIBLE_DEVICES="0"
MODEL_TYPE="bart" # "t5" "bart" "pegasus"
MODEL_NAME="facebook/bart-base" #"t5-large" "facebook/bart-large" "google/pegasus-large"
TRAIN_EXAMPLES="5"
TEST_EXAMPLES="10"
MODE="FINE-TUNING" # "PRIMING" "TRAINING" "FINE-TUNING"
MAX_SEQ_LENGTH="80"

python3 cli_copy.py \
	--method pet \
	--wrapper_type generative \
	--pattern_ids 2 3 4 5 \
	--data_dir . \
	--model_type $MODEL_TYPE \
	--model_name_or_path $MODEL_NAME \
	--task_name cnn-dailymail \
	--output_dir "output-bart-100" \
	--train_examples $TRAIN_EXAMPLES \
	--test_examples 50 \
	--unlabeled_examples 5 \
	--do_eval \
	--learning_rate 1e-4 \
	--eval_set test \
    --overwrite_output_dir \
	--pet_per_gpu_eval_batch_size 2 \
	--pet_per_gpu_train_batch_size 2 \
	--pet_gradient_accumulation_steps 16 \
	--output_max_seq_length ${MAX_SEQ_LENGTH} \
	--pet_max_steps 250 \
	--pet_max_seq_length 218 \
	--sc_per_gpu_train_batch_size 1 \
	--sc_gradient_accumulation_steps 8 \
	--sc_per_gpu_eval_batch_size 1 \
	--sc_max_steps 250 \
	--sc_max_seq_length 218 \
	--optimizer adafactor \
	--epsilon 0.1 \
	--pet_repetitions 1 \
	--train_data_seed 0 \
	--multi_pattern_training \
	--cutoff_percentage 0.2 \


# UNSUPERVISED: To evaluate a pretrained language model with the default PET 
# patterns and verbalizers, but without fine-tuning, remove the argument 
# --do_train and add --no_distillation so that no final distillation is performed.

# PRIMING: If you want to use priming, remove the argument --do_train and add the arguments 
# --priming --no_distillation so that all training examples are used for priming 
# and no final distillation is performed.
# Remember that you may need to increase the maximum sequence length to a much larger value, 
# e.g. --pet_max_seq_length 5000. This only works with language models that support 
# such long sequences, e.g. XLNet. For using XLNet, you can specify -
# -model_type xlnet --model_name_or_path xlnet-large-cased --wrapper_type plm.