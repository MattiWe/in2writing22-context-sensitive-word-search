# in2writing22-neural-netspeak

This is the code for the submission `Language Models as Context-sensitive Word Search Engines`. 

Via cli, this code can reproduce the data used, train the models, do the predictions, and prduce the evaluation resources. 

## Quick Start

1. Setup and activate a venv and install the dependencies 

    ```bash
    ~$ python3 -m venv venv 
    ~$ . venv/bin/activate
    (venv) ~$ pip install wheel
    (venv) ~$ pip install .
    (venv) ~$ python -m spacy download en_core_web_trf
   ```

2. Start the cli and follow the help

   ```bash
   (venv) ~$ main
   ```
   
CLI Options are:
- main data (cloth|netspeak|singlechoice|evaluate) -> do data preparation and evaluation
- main train -> train the models
- main test -> make predictions on the test set
- main evaluate - > compute the evaluation measures on the predictions and output the presentations. 
