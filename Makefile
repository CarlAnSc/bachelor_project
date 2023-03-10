.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

## Include ENV variables in .env file
include .env
#export $(shell sed 's/=.*//' .env)

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = CJ_bachelor
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -r requirements_app.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Build docker image for the app
docker_build_app:
	docker build \
		-f dockerfiles/app.dockerfile \
		-t gcr.io/$(GCP_PROJECT)/bach_app:latest \
		.
## Tag latest
docker_tag_app:
	docker tag bach_app gcr.io/ageless-accord-377810/bach_app

## Push to GCP
docker_push_app:
	docker push \
		gcr.io/$(GCP_PROJECT)/bach_app:latest

## Do all the docker commands
docker_build_push_app: docker_build_app docker_tag_app docker_push_app

## Make Dataset
# data:
# 	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

# ## Run training
# train:
# 	$(PYTHON_INTERPRETER) src/models/train_model.py

# ## Evaluate the model
# evaluate:
# 	$(PYTHON_INTERPRETER) src/models/predict_model.py

# ## Host app locally
# app:
# 	streamlit run app/Upload.py

# ## Pull data from DVC
# data_pull:
# 	dvc pull

# ## Delete all compiled Python files
# clean:
# 	find . -type f -name "*.py[co]" -delete
# 	find . -type d -name "__pycache__" -delete

# ## Format the files
# format:
# 	isort src/ app/ tests/
# 	black src/ app/ tests/

# ## Lint using flake8
# lint:
# 	flake8 src/ app/ tests/

# ## Build docker image for model training
# docker_build_train:
# 	docker build \
# 		-f dockerfiles/train.dockerfile \
# 		-t gcr.io/$(GCP_PROJECT)/train:latest \
# 		.

# ## Build docker image for model evaluation
# docker_build_predict:
# 	docker build \
# 		-f dockerfiles/predict.dockerfile \
# 		-t gcr.io/$(GCP_PROJECT)/predict:latest \
# 		.

# ## Build docker image for the app
# docker_build_app:
# 	docker build \
# 		-f dockerfiles/app.dockerfile \
# 		-t gcr.io/$(GCP_PROJECT)/app:latest \
# 		.

# ## Build docker images for both training, evaluation and app all in one
# docker_build: docker_build_train docker_build_predict docker_build_app

# ## Push training docker image to gcloud container registry
# docker_push_train:
# 	docker push \
# 		gcr.io/$(GCP_PROJECT)/train:latest

# ## Push evaluation docker image to gcloud container registry
# docker_push_predict:
# 	docker push \
# 		gcr.io/$(GCP_PROJECT)/predict:latest

# ## Push app docker image to gcloud container registry
# docker_push_app:
# 	docker push \
# 		gcr.io/$(GCP_PROJECT)/app:latest

# ## Push all docker images to gcloud container registry
# docker_push: docker_push_train docker_push_predict docker_push_app

# ## Run the training docker image
# docker_train:
# 	docker run \
# 		--gpus all \
# 		-it \
# 		--rm \
# 		--ipc=host \
# 		--env-file .env \
# 		gcr.io/$(GCP_PROJECT)/train:latest

# ## Run the training docker image in GCP Vertex AI
# docker_train_cloud:
# 	gcloud ai \
#       --ai \
#       --custom-jobs \
#       --create \
#       --region=europe-west1 \
#       --display-name=train-$(git rev-parse --short HEAD) \
#       --args=++hydra.job.env_set.WANDB_ENTITY=$(WANDB_ENTITY) \
#       --args=++hydra.job.env_set.WANDB_PROJECT=$(WANDB_PROJECT) \
#       --args=++hydra.job.env_set.WANDB_API_KEY=$(WANDB_API_KEY) \
#       --worker-pool-spec=machine-type=n1-standard-8 \
# 	  --container-image-uri=gcr.io/$(GCP_PROJECT)/train:latest

# ## Run the evaluation docker image
# docker_predict:
# 	docker run \
# 		gcr.io/$(GCP_PROJECT)/predict:latest

# ## Run the predict docker image in GCP Vertex AI
# docker_predict_cloud:
# 	gcloud ai \
#       --ai \
#       --custom-jobs \
#       --create \
#       --region=europe-west1 \
#       --display-name=predict-$(git rev-parse --short HEAD) \
#       --args=++hydra.job.env_set.WANDB_ENTITY=$(WANDB_ENTITY) \
#       --args=++hydra.job.env_set.WANDB_PROJECT=$(WANDB_PROJECT) \
#       --args=++hydra.job.env_set.WANDB_API_KEY=$(WANDB_API_KEY) \
#       --worker-pool-spec=machine-type=n1-standard-8 \
# 	  --container-image-uri=gcr.io/$(GCP_PROJECT)/predict:latest

# ## Deploy the app locally from the docker image
# docker_deploy_app_local:
# 	docker run \
# 		-p 8501:8501 \
# 		--env-file .env \
# 		gcr.io/$(GCP_PROJECT)/app:latest

# ## Deploy the app in gcloud Cloud Run from the docker image
# docker_deploy_app_cloud:
# 	gcloud run deploy app \
# 		--image=gcr.io/$(GCP_PROJECT)/app:latest \
# 		--allow-unauthenticated \
# 		--cpu=2 \
# 		--memory=8Gi \
# 		--port=8501 \
# 		--set-env-vars=WANDB_ENTITY=$(WANDB_ENTITY),WANDB_PROJECT=$(WANDB_PROJECT),WANDB_API_KEY=$(WANDB_API_KEY),WANDB_MODELCHECKPOINT=$(WANDB_MODELCHECKPOINT) \
# 		--service-account=$(SERVICE_ACCOUNT) \
# 		--region=europe-west1 \
# 		--project=$(GCP_PROJECT)

# ## Set up python interpreter environment
# create_environment:
# ifeq (True,$(HAS_CONDA))
# 		@echo ">>> Detected conda, creating conda environment."
# ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
# 	conda create -y --name $(PROJECT_NAME) python=3.9
# else
# 	conda create --name $(PROJECT_NAME) python=2.7
# endif
# 		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
# else
# 	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
# 	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
# 	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
# 	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
# 	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
# endif

# ## Test python environment is setup correctly
# test_environment:
# 	$(PYTHON_INTERPRETER) test_environment.py

# #################################################################################
# # PROJECT RULES                                                                 #
# #################################################################################



# #################################################################################
# # Self Documenting Commands                                                     #
# #################################################################################

# .DEFAULT_GOAL := help

# # Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# # sed script explained:
# # /^##/:
# # 	* save line in hold space
# # 	* purge line
# # 	* Loop:
# # 		* append newline + line to hold space
# # 		* go to next line
# # 		* if line starts with doc comment, strip comment character off and loop
# # 	* remove target prerequisites
# # 	* append hold space (+ newline) to line
# # 	* replace newline plus comments by `---`
# # 	* print line
# # Separate expressions are necessary because labels cannot be delimited by
# # semicolon; see <http://stackoverflow.com/a/11799865/1968>
# .PHONY: help
# help:
# 	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
# 	@echo
# 	@sed -n -e "/^## / { \
# 		h; \
# 		s/.*//; \
# 		:doc" \
# 		-e "H; \
# 		n; \
# 		s/^## //; \
# 		t doc" \
# 		-e "s/:.*//; \
# 		G; \
# 		s/\\n## /---/; \
# 		s/\\n/ /g; \
# 		p; \
# 	}" ${MAKEFILE_LIST} \
# 	| LC_ALL='C' sort --ignore-case \
# 	| awk -F '---' \
# 		-v ncol=$$(tput cols) \
# 		-v indent=19 \
# 		-v col_on="$$(tput setaf 6)" \
# 		-v col_off="$$(tput sgr0)" \
# 	'{ \
# 		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
# 		n = split($$2, words, " "); \
# 		line_length = ncol - indent; \
# 		for (i = 1; i <= n; i++) { \
# 			line_length -= length(words[i]) + 1; \
# 			if (line_length <= 0) { \
# 				line_length = ncol - indent - length(words[i]) - 1; \
# 				printf "\n%*s ", -indent, " "; \
# 			} \
# 			printf "%s ", words[i]; \
# 		} \
# 		printf "\n"; \
# 	}' \
# 	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
