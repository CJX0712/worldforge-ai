PY ?= python

.PHONY: help verify test smoke baseline bench serve lint clean install

help:
	@echo "make install   安装开发依赖（editable + dev extras）"
	@echo "make verify    六阶段门禁：字符门禁 -> pytest -> 导入 -> 冒烟 -> 评估门限 -> HTTP E2E"
	@echo "make test      只跑 pytest"
	@echo "make smoke     只跑 HTTP 端到端冒烟"
	@echo "make baseline  重录评估基线"
	@echo "make bench     跑吞吐/延迟基准"
	@echo "make serve     启动 HTTP 网关"
	@echo "make lint      ruff 静态检查"

install:
	$(PY) -m pip install -e ".[dev]"

verify:
	$(PY) scripts/verify.py

test:
	$(PY) -m pytest tests -q

smoke:
	$(PY) scripts/smoke.py

baseline:
	$(PY) scripts/record_baseline.py --runs 5

bench:
	$(PY) scripts/bench.py

serve:
	$(PY) -m worldforge.serving.app

lint:
	$(PY) -m ruff check src tests scripts

clean:
	$(PY) -c "import glob,os,shutil; [shutil.rmtree(d) for d in glob.glob('**/__pycache__', recursive=True)]"
