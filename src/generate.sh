python create_ubuntu_dataset.py "$@" --output 'train.csv' 'train' --examples 100000 # appropriate number for refrigerator dataset
python create_ubuntu_dataset.py "$@" --output 'test.csv' 'test'
python create_ubuntu_dataset.py "$@" --output 'valid.csv' 'valid'
