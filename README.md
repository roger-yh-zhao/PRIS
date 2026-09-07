PDIScore
===
The paper is accessible at https://www.nature.com/articles/s41401-025-01688-3
<img src="https://github.com/user-attachments/assets/aabe1de7-f2ff-4161-a014-e3b7dc8ca060" width="500px">


Build environment
-------
````
conda create --prefix xxx --file ./requirements_conda.txt      
pip install -r ./requirements_pip.txt
````

Example
-------
Generate pocket:
````
cd test
python pocket.py --name 1qne
````

Reorder the residue/nucleotide of protein/nucleic acid:
````
python reorder.py --name 1qne
````

Convert pdb structure to graph:
````
python pdb2graph.py -idf ids.txt
````

Output the final score:
````
python test.py -of score.csv
````
