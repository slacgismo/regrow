# Setup

## First time

You must install the app from the root directory.

~~~
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install pip --upgrade -r examples/single-node/requirements.txt
~~~

## Returning to work

~~~
source .venv/bin/activate
~~~

# Running

You must run/edit the app from the root directory.

~~~
marimo run examples/single-node/single_node.py
~~~
