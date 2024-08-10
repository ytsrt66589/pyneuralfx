# Training 

## Set the configuration file 

Modify the path to the configuration file you aim to use 
```python 
'''in main_full.py or main_snapshot.py'''

# ============================================================ #
# Config 
# ============================================================ #
# Load config from yaml files 
cmd = {
    'config': './configs/rnn/gru/concat_gru.yml'
}

args = utils.load_config(cmd['config'])
print(' > config:', cmd['config'])
```

## Run the training code 

Run 
```python
    python3 main_snapshot.py 
```
for snapshot modeling 

---

Run 
```python
    python3 main_full.py 
```
for full modeling 

---