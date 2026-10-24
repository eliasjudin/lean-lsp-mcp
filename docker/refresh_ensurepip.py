"""Refresh the bundled pip used when Python creates future virtual environments."""

import argparse
import email
import re
import shutil
import zipfile
from pathlib import Path


def refresh(wheel: Path, ensurepip_dir: Path, version: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        metadata = email.message_from_bytes(
            archive.read(f"pip-{version}.dist-info/METADATA")
        )
        if metadata["Name"] != "pip" or metadata["Version"] != version:
            raise ValueError("Unexpected pip wheel identity")
        certificates = archive.read("pip/_vendor/certifi/cacert.pem")
        if b"emSign ECC Root CA - G3" in certificates:
            raise ValueError("pip wheel retains the compromised CA trust path")

    source = ensurepip_dir / "__init__.py"
    updated, count = re.subn(
        r"(?m)^_PIP_VERSION = ['\"][^'\"]+['\"]$",
        f"_PIP_VERSION = {version!r}",
        source.read_text(),
    )
    if count != 1:
        raise ValueError("Unrecognized ensurepip version declaration")
    bundled = ensurepip_dir / "_bundled"
    target = bundled / wheel.name
    shutil.copyfile(wheel, target)
    source.write_text(updated)
    for previous in bundled.glob("pip-*.whl"):
        if previous != target:
            previous.unlink()


if __name__ == "__main__":
    import ensurepip

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("version")
    args = parser.parse_args()
    refresh(args.wheel, Path(ensurepip.__file__).parent, args.version)
