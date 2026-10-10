.PHONY: install run train test lint docker
install:   ; pip install -r requirements-dev.txt
run:       ; streamlit run app.py
train:     ; python scripts/train.py
test:      ; pytest
lint:      ; ruff check .
docker:    ; docker build -t maritime-intel . && docker run --rm -p 8501:8501 maritime-intel
