import fire
import torch


def main(model: str, output_path: str):
    cpt = torch.load(model)
    weights = cpt["model"]
    weights.pop("encoder.version")
    weights.pop("decoder.version")
    torch.save(weights, output_path)


if __name__ == '__main__':
    fire.Fire(main)
