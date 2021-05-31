cd neuralcoref
pip install -r requirements.txt
pip install -e .
python -m spacy download en
cd ../
python main.py