export BERT_BASE_DIR=$HOME/work/valence_tweet/uncased_L-12_H-768_A-12/
export MY_DATASET=$HOME/work/valence_tweet/data/


SINGULARITYENV_BERT_BASE_DIR=$HOME/work/valence_tweet/uncased_L-12_H-768_A-12/
SINGULARITYENV_MY_DATASET=$HOME/work/valence_tweet/data/

python run_classifier.py \
  --task_name=valence \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$HOME/work/valence_tweet/data/ \
  --vocab_file=$HOME/work/valence_tweet/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=$HOME/work/valence_tweet/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=$HOME/work/valence_tweet/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=64 \
  --learning_rate=5e-5 \
  --num_train_epochs=5.0 \
  --output_dir=/home/u8/liuhao16/work/valence_tweet/output/

