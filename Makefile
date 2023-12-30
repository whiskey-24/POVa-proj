.PHONY: venv run zip

ARCHIVE_NAME = team_xvisku01.zip
VENV_NAME = venv
PYTHON = python3

venv:
	@{ \
  	if [ ! -d "venv" ]; \
  	then \
    	$(PYTHON) -m venv $(VENV_NAME); \
    	source $(VENV_NAME)/bin/activate; \
    	pip3 install -r requirements.txt; \
  	fi \
	}

run: venv
	@{ \
    source $(VENV_NAME)/bin/activate; \
 	$(PYTHON) application.py; \
 	}

zip:
	find detector/yolov8 detector/misc detector/retinaface/Pytorch_Retinaface -type f ! -name "*.pth" ! -name "*.pt" ! -name "*.zip" -print | zip $(ARCHIVE_NAME) -@
	find detector/retinaface/evaluate_retinaface.py -type f -print | zip $(ARCHIVE_NAME) -@
	find map_reg/ -type f ! -name "*.jpg" ! -name "*.jpeg" ! -name "*.png" ! -name "*.mp4" ! -name "*.pt" ! -name "*.info" ! -name "*.txt" ! -name "*.pth" -print | zip $(ARCHIVE_NAME) -@