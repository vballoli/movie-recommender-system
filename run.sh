pip3 install -r requirements.txt
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip -P preprocessing/
unzip preprocessing/ml-100k.zip -d preprocessing/ml-100k
python3 app.py
open docs/_build/html/py-modindex.html
