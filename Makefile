# Makefile

VENV_DIR := .venv

REPO_HTML := https://github.com/aapetrakova/optics_4_sem.git
REPO_DIR := optics_4_sem

MAIN_FILE := $(REPO_DIR)/application/main.py

REQUIREMENTS := pyqt6 matplotlib numpy scipy torch

.PHONY: all clone setup run clean

all: run

clone:
  @if [ ! -d "$(REPO_DIR)" ]; then \
    echo "Клонируем репозиторий..."; \
    git clone $(REPO_HTML); \
  fi
  @cd $(REPO_DIR) && git checkout main

setup: clone
  @if [ ! -d "$(VENV_DIR)" ]; then \
    echo "Создаем виртуальное окружение..."; \
    python3 -m venv $(VENV_DIR); \
  fi
  @echo "Устанавливаем зависимости..."; \
  . $(VENV_DIR)/bin/activate && pip install --upgrade pip && pip install $(REQUIREMENTS)

run: setup
  @echo "Запускаем приложение..."; \
  . $(VENV_DIR)/bin/activate && python $(MAIN_FILE)

clean:
  rm -rf $(VENV_DIR) $(REPO_DIR)
