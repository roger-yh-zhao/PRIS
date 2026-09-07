
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

Convert pdb structure to graph:
````
python pdb2graph.py
````

Output the pwm:
````
python test_model.py -m weight_MS2.pth -featsl 27 -e gps -gps_local ASGAT -gps_global SparseAttention
````

Convert pwm to score
````
python pwm2score.py
````

calculate metrics
````
python metric.py
````
