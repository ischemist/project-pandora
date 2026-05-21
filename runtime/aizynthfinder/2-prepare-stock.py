"""Convert plain SMILES stock files to AiZynthFinder HDF5 stock files."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Iterable

import pandas as pd
from aizynthfinder.chem import Molecule, MoleculeException


def _get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("smiles2stock")
    parser.add_argument("--files", required=True, nargs="+", help="files containing SMILES")
    parser.add_argument(
        "--source",
        choices=["plain", "module"],
        default="plain",
        help="read SMILES from plain files or a python module",
    )
    parser.add_argument("--output", required=True, default="", help="output filename or source tag")
    return parser.parse_args()


def _convert_smiles(smiles_list: Iterable[str]) -> Iterable[str]:
    for smiles in smiles_list:
        try:
            yield Molecule(smiles=smiles, sanitize=True).inchi_key
        except MoleculeException:
            print(
                f"Failed to convert {smiles} to inchi key. Probably due to sanitation.",
                flush=True,
            )


def extract_plain_smiles(files: list[str]) -> Iterable[str]:
    for filename in files:
        print(f"Processing {filename}", flush=True)
        with open(filename, encoding="utf-8") as fileobj:
            for line in fileobj:
                yield line.strip()


def extract_smiles_from_module(files: list[str]) -> Iterable[str]:
    module_name = files.pop(0)
    module = importlib.import_module(module_name)
    if not files:
        for smiles in module.extract_smiles():  # type: ignore[attr-defined]
            yield smiles
    else:
        for filename in files:
            print(f"Processing {filename}", flush=True)
            for smiles in module.extract_smiles(filename):  # type: ignore[attr-defined]
                yield smiles


def make_hdf5_stock(inchi_keys: Iterable[str], filename: str) -> None:
    data = pd.DataFrame.from_dict({"inchi_key": inchi_keys})
    data = data.drop_duplicates("inchi_key")
    data.to_hdf(filename, "table")
    print(f"Created HDF5 stock with {len(data)} unique compounds")


def main() -> None:
    args = _get_arguments()
    smiles_gen = extract_plain_smiles(args.files) if args.source == "plain" else extract_smiles_from_module(args.files)
    inchi_keys_gen = (inchi_key for inchi_key in _convert_smiles(smiles_gen))
    make_hdf5_stock(inchi_keys_gen, args.output)


if __name__ == "__main__":
    main()
