FROM ubuntu:16.04

# settings
ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

RUN apt-get update \
    # Install runtime dependencies
    && apt-get install -y openjdk-8-jdk libleveldb-dev ant unzip wget curl cpanminus git graphviz maven subversion python vim \
    # Install Python 3
    && apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python3 \
    && pip3 install --upgrade pip

WORKDIR /soar

COPY . .
CMD ["bash"]
