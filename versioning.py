import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from huggingface_hub import HfApi, Repository
from torch import nn

from logger import logger
from utils import markdown_table, uid_from_dictionary


@dataclass
class RepoPath:
    root: Path

    def __post_init__(self):
        self.root.mkdir(exist_ok=True, parents=True)
        self.params: Path = self.root / "params.json"
        self.model_card: Path = self.root / "README.md"


class ModelVersioningHandler:
    def __init__(self, repo_id: str, local_dir: Path):
        self.repo_id = repo_id
        self.hf_api = HfApi()
        self.repo_path = RepoPath(root=local_dir)
        self.repo = self.maybe_create_repo()

    def maybe_create_repo(self) -> Repository:
        self.hf_api.create_repo(self.repo_id, exist_ok=True)
        return Repository(local_dir=str(self.repo_path.root), clone_from=self.repo_id)

    def has_params_changed(self, params: Dict[str, Any]) -> bool:
        # load the current params
        has_changed: bool = True
        # if params exist on disk, open it
        if self.repo_path.params.exists():
            with self.repo_path.params.open("r") as f:
                params_from_disk = json.load(f)
                # this is a weak check, works only with primitive types and non nested dictionaries
                has_changed = params_from_disk != params

        return has_changed

    def maybe_switch_branch(self, params: Dict[str, Any], repo: Repository):
        # create a uid from params
        branch_name: str = uid_from_dictionary(params)
        if branch_name != repo.current_branch:
            # create a new branch
            repo.git_checkout(branch_name, create_branch_ok=True)
        try:
            repo.git_pull()
        except OSError:
            # pull fails if the branch was not pushed to origin
            pass

    def add_params(self, params: Dict[str, Any]):
        with self.repo_path.params.open("w") as f:
            json.dump(params, f)

    def add_model_card(self, global_params: Dict[str, Dict[str, Any]]):
        with self.repo_path.model_card.open("w") as f:
            uids = global_params.keys()
            params = global_params.values()
            # add `uid` for each parameter dict
            params_with_uids = [
                {"uid": uid, **param} for uid, param in zip(uids, params)
            ]
            df = pd.DataFrame.from_records(params_with_uids)
            # add links to the branches!
            df["uid"] = df["uid"].apply(
                lambda x: f"[{x}]({self.hf_api.endpoint}/{self.repo_id}/tree/{x})"
            )

            f.write(df.to_markdown())

    def update_global_params(self, params: Dict[str, Any], repo: Repository):
        branch_name = repo.current_branch
        # switch to main
        repo.git_checkout("main")
        repo.git_pull()
        # we want to create a list of dictionary using all the previous stored parameters and the new ones (`params`)
        global_params = {}
        if self.repo_path.params.exists():
            with self.repo_path.params.open("r") as f:
                global_params = json.load(f)
        # update with the new ones
        global_params[branch_name] = params
        with self.repo_path.params.open("w") as f:
            json.dump(global_params, f)
        # create a README.md with a list of all the model's versions
        self.add_reference_model_card(global_params)
        repo.push_to_hub("Updated params.json")

    def save_model(self, model: nn.Module):
        state_dict = model.state_dict()
        torch.save(state_dict, self.repo_path.root / "model.pth")

    def __call__(self, params: Dict[str, Any], model: nn.Module):
        """
        Versions a model given a set of parameters. The parameters are hashed and used as unique identifier. Based on the hash, a branch in created and the model's weights are added to it.

        Different model runs with the same parameters will result in multiple commits on the same branch.

        The `main` branch keeps an overview of all the models' versions.

        Args:
            params (Dict[str, Any]): A set of parameters we want to track.
            model (nn.Module): A model we want to version.
        """
        has_params_changed: bool = self.has_params_changed(params)

        if has_params_changed:
            logger.info("Parameters have beend changed.")
            self.maybe_switch_branch(params, self.repo)
            self.add_model_card(params, self.repo)
            self.repo.push_to_hub("Added parameters.")

        self.save_model(model)
        self.repo.push_to_hub("Update model weights.")

        if has_params_changed:
            logger.info("Updating params in `main`.")
            self.update_global_params(params, self.repo)

        logger.info(
            f"See all the model's versions at {self.hf_api.endpoint}/{self.repo_id}"
        )

    def delete(self, also_from_hub: bool = False):
        """
        Deletes the versioning on disk and on the hub is `also_from_hub` is True/

        Args:
            also_from_hub (bool, optional): If `True`, the versioning on the hub is deleted. Defaults to False.
        """

        shutil.rmtree(self.repo_path.root)
        if also_from_hub:
            logger.info(f"Removing repo at {self.repo_id}")
            self.hf_api.delete_repo(self.repo_id)
