# Mode versioning with hugging face hub!
**WIP** This repo is a work in progress, so bugs are expected :)

What if we use a git-based cloud hosting serving (like [hf hub](https://huggingface.co/models)) to version our models?

## How it works
Model versioning means keeping track of a set of `parameters` and a `model` obtained using them. 

So what if, each set of parameters defines a `branch` and each model trained using that specific set is a `commit` on that branch?

We can keep the `main` one untouched and use it to list all the branches with their parameters.

## Example

Let's see an example (copied from `./example.py`).

```python
from pathlib import Path

from model import BoringModel
from versioning import ModelVersioningHandler

# define some parameters
all_params = [{"hidden_size": 8}, {"hidden_size": 16}, {"hidden_size": 32}]

# create our model versioning handler
model_versioning_handler = ModelVersioningHandler(
    local_dir=Path("./test"), repo_id="zuppif/versioning-test"
)

for params in all_params:
    # here we may want to train the model
    model = BoringModel(**params)
    # our handler will push and keep track of both params and model's weights
    model_versioning_handler(params, model)
```

The above code has three sets of parameters:

- `{"hidden_size": 8}`
- `{"hidden_size": 16}`
- `{"hidden_size": 32}`

These parameters are used to create `BoringModel`, we don't care about it, just keep in mind it's a pytorch model

So, to version our model, we will create 3 branches, one for each set of parameters, + `main`. Every time we change our model's weight but we use the same parameter, we will push on the same branch.

So, the `main` branch keeps a list of all the other branches

![alt](/images/main.png)

A `branch` is identified using the current parameters hash, it contains the model's weights and the parameters

![alt](/images/branch.png)

Let's see inside `params.json`

![alt](/images/branch_params.png)

Different runs with the same parameters will commit the model's weight to the same branch, making it easier to version it.

![alt](/images/branch_commits.png)
