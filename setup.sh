conda create -n quote -y python=3.8
conda activate 
cd neuralcoref
pip install -r requirements.txt
pip install -e .
python -m spacy download en
echo downloading wikidata......
python -m spacyEntityLinker "download_knowledge_base"
cd ../
