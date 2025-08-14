import pathlib

import fire
import torch
from safetensors import safe_open


def main(origin: str, model: str, output_path: str, prefix: str = "nllb." ):

    origin = torch.load(origin)

    for file in pathlib.Path(model).glob(f"*.safetensors"):
        with safe_open(file, framework="pt") as merged:
            for name in merged.keys():
                weight = merged.get_tensor(name)
                assert prefix + name in origin["state_dict"], name
                origin["state_dict"][prefix + name] = weight

    torch.save(origin, output_path)


if __name__ == '__main__':
    fire.Fire(main)
