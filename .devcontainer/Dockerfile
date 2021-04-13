FROM mcr.microsoft.com/vscode/devcontainers/miniconda:latest

COPY ci/*.txt /tmp/conda-tmp/
RUN sed -i -e "s/scipy==.*/scipy==1.5.3/" /tmp/conda-tmp/requirements.txt
RUN /opt/conda/bin/conda config --add channels conda-forge
RUN /opt/conda/bin/conda config --set channel_priority strict
RUN /opt/conda/bin/conda install --quiet --yes --file /tmp/conda-tmp/test_requirements.txt --file /tmp/conda-tmp/extra_requirements.txt --file /tmp/conda-tmp/requirements.txt --file /tmp/conda-tmp/linting_requirements.txt --file /tmp/conda-tmp/doc_requirements.txt