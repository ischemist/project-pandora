"""Convert plain SMILES stock files to AiZynthFinder HDF5 stock files."""

import argparse
from pathlib import Path

import pandas as pd
from aizynthfinder.chem import Molecule, MoleculeException


def main() -> None:
    parser = argparse.ArgumentParser("prepare-stock")
    parser.add_argument("--files", required=True, nargs="+", type=Path, help="plain text files containing SMILES")
    parser.add_argument("--output", required=True, type=Path, help="output HDF5 stock file")
    args = parser.parse_args()

    inchi_keys = []
    for path in args.files:
        print(f"Processing {path}", flush=True)
        with path.open(encoding="utf-8") as fileobj:
            for line in fileobj:
                smiles = line.strip()
                if not smiles:
                    continue

                try:
                    inchi_keys.append(Molecule(smiles=smiles, sanitize=True).inchi_key)
                except MoleculeException:
                    print(f"Failed to convert {smiles} to inchi key. Probably due to sanitation.", flush=True)

    data = pd.DataFrame({"inchi_key": inchi_keys}).drop_duplicates("inchi_key")
    data.to_hdf(args.output, key="table")
    print(f"Created HDF5 stock with {len(data)} unique compounds")


if __name__ == "__main__":
    main()
