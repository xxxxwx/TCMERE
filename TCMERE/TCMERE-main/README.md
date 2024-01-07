# A Joint Entity Relation Extraction Method for Document Level Traditional Chinese Medicine Texts

##  Instructions
The code has been tested with Python 3. To install the dependencies, please run:
```
pip install -r requirements.txt
```

After downloading the datasets, please create a new folder `resources` and put the datasets into that folder.
Overall, the folder structure of the entire repo should look like:
```
...
models/
resources/
--- biorelex/
------- train.json
------- dev.json
scorer/
.gitignore

...
```
./TCMroberta
In this folder are the secondary pre-training language models in the field of TCM

For training, please refer to the scripts  `trainer.py`. For example, to train a basic model for TCMERE, you can simply run:
```
python trainer.py
```

There are some redundant code in this repo.Some of the code retains the source code names. I am going to remove them soon.
