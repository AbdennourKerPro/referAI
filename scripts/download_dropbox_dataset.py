#!/usr/bin/env python3
"""Télécharge et extrait un dossier partagé Dropbox.

Compatible avec Python 3.8+ et sans dépendance externe.
"""

import argparse
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


DEFAULT_URL = (
    "https://www.dropbox.com/scl/fo/6a9p1ukbyjvb0gderu2ng/"
    "AEtiBYzDCt0qLdGOmYi9nAI?rlkey=lbpr7s4106n6y38gdky2oq6ch&dl=0"
)
CHUNK_SIZE = 1024 * 1024


def direct_download_url(shared_url: str) -> str:
    """Remplace proprement le paramètre Dropbox ``dl`` par ``1``."""
    parsed = urllib.parse.urlsplit(shared_url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query = [(key, value) for key, value in query if key != "dl"]
    query.append(("dl", "1"))
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), parsed.fragment)
    )


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("o", "Kio", "Mio", "Gio", "Tio"):
        if value < 1024 or unit == "Tio":
            return "{:.1f} {}".format(value, unit)
        value /= 1024
    return "{} o".format(size)


def download(url: str, output: Path, retries: int = 3) -> None:
    request = urllib.request.Request(
        direct_download_url(url),
        headers={"User-Agent": "referAI-dataset-downloader/1.0"},
    )

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, output.open("wb") as target:
                total_header: Optional[str] = response.headers.get("Content-Length")
                total = int(total_header) if total_header and total_header.isdigit() else 0
                downloaded = 0
                last_update = 0.0

                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    target.write(chunk)
                    downloaded += len(chunk)

                    now = time.monotonic()
                    if now - last_update >= 0.2:
                        if total:
                            percent = min(downloaded * 100 / total, 100)
                            status = "\rTéléchargement : {:5.1f}% ({}/{})".format(
                                percent, human_size(downloaded), human_size(total)
                            )
                        else:
                            status = "\rTéléchargement : {}".format(human_size(downloaded))
                        print(status, end="", flush=True)
                        last_update = now

                print("\rTéléchargement terminé : {}                 ".format(human_size(downloaded)))
                return
        except (OSError, urllib.error.URLError) as error:
            output.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(
                    "échec du téléchargement après {} tentative(s) : {}".format(retries, error)
                ) from error
            delay = 2 ** (attempt - 1)
            print(
                "Tentative {}/{} échouée ({}). Nouvel essai dans {} s...".format(
                    attempt, retries, error, delay
                ),
                file=sys.stderr,
            )
            time.sleep(delay)


def validate_zip_members(archive: zipfile.ZipFile, destination: Path) -> None:
    """Refuse les entrées ZIP qui sortiraient du dossier de destination."""
    root = destination.resolve()
    for member in archive.infolist():
        member_path = (root / member.filename).resolve()
        try:
            member_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                "archive non sûre : le chemin {!r} sort du dossier cible".format(member.filename)
            ) from error


def extract(archive_path: Path, staging: Path) -> None:
    if not zipfile.is_zipfile(str(archive_path)):
        raise RuntimeError(
            "Dropbox n'a pas renvoyé une archive ZIP. Vérifiez que le lien est public "
            "et que son téléchargement est autorisé."
        )

    with zipfile.ZipFile(str(archive_path)) as archive:
        validate_zip_members(archive, staging)
        members = archive.infolist()
        for index, member in enumerate(members, start=1):
            archive.extract(member, str(staging))
            print(
                "\rExtraction : {}/{} fichiers".format(index, len(members)),
                end="",
                flush=True,
            )
        print()


def install_staging(staging: Path, destination: Path, force: bool) -> None:
    if not destination.exists():
        staging.replace(destination)
        return

    if not destination.is_dir():
        raise RuntimeError("la destination existe mais n'est pas un dossier : {}".format(destination))

    if any(destination.iterdir()) and not force:
        raise RuntimeError(
            "la destination n'est pas vide : {} (utilisez --force pour fusionner)".format(
                destination
            )
        )

    shutil.copytree(str(staging), str(destination), dirs_exist_ok=True)


def validate_destination(destination: Path, force: bool) -> None:
    """Détecte les conflits avant de lancer un téléchargement potentiellement long."""
    if not destination.exists():
        return
    if not destination.is_dir():
        raise RuntimeError("la destination existe mais n'est pas un dossier : {}".format(destination))
    if any(destination.iterdir()) and not force:
        raise RuntimeError(
            "la destination n'est pas vide : {} (utilisez --force pour fusionner)".format(
                destination
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Télécharge le dataset Dropbox et l'extrait dans le dossier demandé."
    )
    parser.add_argument("destination", type=Path, help="dossier dans lequel extraire le dataset")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="lien de partage Dropbox (le lien du dataset est utilisé par défaut)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="fusionne avec une destination non vide et remplace les fichiers homonymes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.destination.expanduser().resolve()
    parent = destination.parent

    try:
        validate_destination(destination, args.force)
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="referai-download-", dir=str(parent)) as temp_dir:
            temp_root = Path(temp_dir)
            archive_path = temp_root / "dataset.zip"
            staging = temp_root / "extracted"
            staging.mkdir()

            print("Destination : {}".format(destination))
            download(args.url, archive_path)
            extract(archive_path, staging)
            install_staging(staging, destination, args.force)
    except (RuntimeError, OSError, zipfile.BadZipFile) as error:
        print("Erreur : {}".format(error), file=sys.stderr)
        return 1

    print("Dataset disponible dans : {}".format(destination))
    return 0


if __name__ == "__main__":
    sys.exit(main())
