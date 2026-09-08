
===


<img src="https://github.com/user-attachments/assets/783cf83c-c298-4be5-861f-71195a055b8b" width="500px">


Build environment
-------
````
# pri.tar.gz is available at https://doi.org/10.5281/zenodo.22651663
mv pri.tar.gz yourpath/anaconda3/envs/pri/pri.tar.gz  
cd yourpath/anaconda3/envs/pri
tar -xzvf pri.tar.gz
conda activate pri
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

Convert pwm to score:
````
python pwm2score.py
````

Calculate metrics:
````
python metric.py
````
