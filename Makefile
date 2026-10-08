.PHONY: env env-cuda install run-mock run-api test test-one clean

env:
	conda env create -f environment.yml
	@test -f .env || cp .env.example .env
	@echo "✅ Run: conda activate model-design"

env-cuda:
	conda env create -f environment-cuda.yml
	@test -f .env || cp .env.example .env
	@echo "✅ Run: conda activate model-design (CUDA)"

install: env

run-mock:
	./run.sh --mock

run-api:
	./run.sh --api-key $(API_KEY)

test:
	bash -c 'source $$(conda info --base)/etc/profile.d/conda.sh && conda activate model-design && python -m pytest tests/ -v'

test-one:
	bash -c 'source $$(conda info --base)/etc/profile.d/conda.sh && conda activate model-design && python -m pytest $(T) -v'

clean:
	rm -rf outputs/*
	conda env remove -n model-design -y
