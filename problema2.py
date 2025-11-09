from pathlib import Path


def main():
    try:
        BASE_DIR = Path(__file__).parent
    except NameError:
        BASE_DIR = Path.cwd()

    imagenes_path = BASE_DIR / "imagenes"
    patentes = sorted(p for p in imagenes_path.glob("img*.png"))

    imagenes_patentes = []


if __name__ == "__main__":
    main()
