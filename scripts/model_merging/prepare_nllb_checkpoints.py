import fire
import torch


def main(model: str, output_path: str, prefix: str = "nllb."):
    cpt = torch.load(model)

    weights = {}
    for name, weight in cpt["state_dict"].items():
        weights[name.replace(prefix, "")] = weight

    torch.save(weights, output_path)


if __name__ == '__main__':
    fire.Fire(main)
