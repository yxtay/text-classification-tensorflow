#!/bin/bash
# download, prepare data
python acl_imdb.py

# fasttext
python fasttext.py --fasttext-bin $HOME/git/fastText/fasttext
