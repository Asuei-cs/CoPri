# CoPri: A Collective Paradigm with Prioritized Sequential Modeling for Cross-domain Recommendation


### Environment
We provide the complete environment configuration in the 'environment.yaml' file, with the key packages summarized as follows：
- python 3.13.3
- pytorch 2.6.0
- loguru 0.7.2
- tqdm 4.67.1
- yaml 0.2.5
- pandas 2.2.3
- optuna 4.3.0

### Datasets
As the dataset files are too large, we host them separately at the following link.
https://www.alipan.com/s/g9kd6VU93YB


### Usage
To run the code, please use the following command format:
```bash
# Rees46 dataset
python main.py --model=CoPri -td=computers -sd=appliances  --hidden_units=128  --t_num_blocks=3 --s_num_blocks=2 --w1=0.3 --w2=0.5   --parent_dir=Data/Rees46/ --datasets=Rees46 
python main.py --model=CoPri -td=computers -sd=electronics   --hidden_units=128  --t_num_blocks=2 --s_num_blocks=2 --w1=0.3  --w2=0.3   --parent_dir=Data/Rees46/ --datasets=Rees46  

python main.py --model=CoPri -td=appliances -sd=computers   --hidden_units=128  --t_num_blocks=1 --s_num_blocks=1 --w1=1.0 --w2=0.3   --parent_dir=Data/Rees46/ --datasets=Rees46  
python main.py --model=CoPri -td=appliances -sd=electronics   --hidden_units=128 --t_num_blocks=3 --s_num_blocks=1 --w1=0.3  --w2=0.3    --parent_dir=Data/Rees46/ --datasets=Rees46  

python main.py --model=CoPri -td=electronics -sd=computers   --hidden_units=128 --t_num_blocks=3 --s_num_blocks=1 --w1=0.3 --w2=0.5  --parent_dir=Data/Rees46/ --datasets=Rees46  
python main.py --model=CoPri -td=electronics -sd=appliances   --hidden_units=128 --t_num_blocks=3 --s_num_blocks=2 --w1=0.1 --w2=0.3   --parent_dir=Data/Rees46/ --datasets=Rees46 

# Amazon dataset
python main.py --model=CoPri -td=Cell -sd=Elec    --hidden_units=64  --t_num_blocks=3 --s_num_blocks=1 --w1=0.5 --w2=0.5   --parent_dir=Data/Amazon/ --datasets=Amazon  
python main.py --model=CoPri -td=Elec -sd=Cell    --hidden_units=64  --t_num_blocks=2 --s_num_blocks=1 --w1=0.7 --w2=0.5   --parent_dir=Data/Amazon/ --datasets=Amazon 

python main.py --model=CoPri -td=Food -sd=Kitchen   --hidden_units=128  --t_num_blocks=2  --s_num_blocks=3 --w1=0.5 --w2=0.9  --parent_dir=Data/Amazon/ --datasets=Amazon 
python main.py --model=CoPri -td=Kitchen -sd=Food   --hidden_units=128 --t_num_blocks=3 --s_num_blocks=2 --w1=0.3 --w2=0.3  --parent_dir=Data/Amazon/ --datasets=Amazon  
```
