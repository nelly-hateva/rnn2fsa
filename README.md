# RRN2FSA

## Prerequisites

Install graphviz

```
sudo apt-get install graphviz
```

Setup virtual environment

````
mkdir ~/.venv
python3 -m venv ~/.venv/tardis
source ~/.venv/tardis/bin/activate
pip3 install -r requirements.txt
````

## Tests

Run the tests with

```
source ~/.venv/tardis/bin/activate
python3 -m unittest
```
