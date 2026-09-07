
===


<img src="https://github.com/user-attachments/assets/783cf83c-c298-4be5-861f-71195a055b8b" width="500px">


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
