cd neuralcoref
pip install -r requirements.txt
pip install -e .
python -m spacy download en
python -m spacyEntityLinker "download_knowledge_base"
cd ../
python main.py