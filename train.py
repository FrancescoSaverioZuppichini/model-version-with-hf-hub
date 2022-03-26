from pathlib import Path

from model import BoringModel
from versioning import ModelVersioningHandler

# define some paramters
all_params = [{"hidden_size": 8}, {"hidden_size": 16}, {"hidden_size": 32}]

# create our model versioning handler
model_versioning_handler = ModelVersioningHandler(
    local_dir=Path("./test"), repo_id="zuppif/versioning-test"
)
# let's remove the local folder
model_versioning_handler.delete()

for params in all_params:
    # here we may want to train the model
    model = BoringModel(**params)
    # our handler will push and keep track both params and model's weights
    model_versioning_handler(params, model)
