
NUM_WORDS=50001
# OUTPUT_DIR="TFRecords/BC-UMBC"
# VOCAB_FILE="dictionaries/BC-UMBC/dictionary.txt"
# TOKENIZED_FILES="BookCorpus/txt_tokenized/*,UMBC/*.txt"
OUTPUT_DIR="TFRecords/BC_filter_empty"
VOCAB_FILE="dictionaries/BC/dictionary.txt"
TOKENIZED_FILES="BookCorpus/txt_tokenized/*.txt"
# OUTPUT_DIR="TFRecords/BC_twitter"
# VOCAB_FILE="dictionaries/BC/dictionary.txt"
# TOKENIZED_FILES="/data/vision/torralba/datasets/books/homemade_tongzhou/twitter_big/*.txt"
# OUTPUT_DIR="TFRecords/UMBC"
# VOCAB_FILE="dictionaries/BC-UMBC/dictionary.txt"
# TOKENIZED_FILES="UMBC/*.txt"

python src/data/preprocess_dataset.py \
  --input_files "$TOKENIZED_FILES" \
  --vocab_file $VOCAB_FILE \
  --output_dir $OUTPUT_DIR \
  --num_words $NUM_WORDS \
  --max_sentence_length 50 \
  --case_sensitive
