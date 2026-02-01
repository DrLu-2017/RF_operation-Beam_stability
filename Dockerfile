FROM gitlab-registry.synchrotron-soleil.fr/pa/collective-effects/mbtrack2:develop
LABEL name albums
USER dockeruser
WORKDIR '/home/dockeruser'

RUN /home/dockeruser/venv/bin/pip3 install --no-cache-dir matplotlib scipy sh pandas tk notebook mpmath numexpr mathphys

# for pycolleff
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN sudo -E apt-get update

WORKDIR '/usr/bin'
RUN sudo -E ln -s pip3 pip-sirius
RUN sudo -E ln -s python3 python-sirius

WORKDIR '/home/dockeruser'
RUN git clone https://github.com/lnls-fac/collective_effects.git
WORKDIR '/home/dockeruser/collective_effects/pycolleff'
RUN sudo -E make develop-install

WORKDIR '/home/dockeruser'
RUN mv collective_effects/pycolleff/pycolleff .

COPY --chown=dockeruser albums /home/dockeruser/albums
ENV PYTHONPATH=/home/dockeruser/
