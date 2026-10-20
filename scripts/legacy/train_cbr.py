"""Entrena el clasificador CBR (Módulo 3) sin necesidad de `-m`.

Wrapper de `triager_omega.cbr.train`: toda la lógica vive en el paquete; este
script solo permite ejecutarlo por ruta de archivo (como el resto de scripts/).
Acepta los mismos argumentos que el módulo.

Ejecutar:
    uv run python scripts/train_cbr.py                      # modo 'both' (crudo+destilado)
    uv run python scripts/train_cbr.py --text-mode raw      # ablación §11.2.4
    uv run python scripts/train_cbr.py --batch-size 8       # menos memoria en MPS
"""

from dotenv import load_dotenv

load_dotenv()  # carga HF_TOKEN (y demás claves) del .env al entorno

from triager_omega.cbr.train import main

if __name__ == "__main__":
    main()
