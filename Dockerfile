FROM qgis/qgis

# Set environment variables for QGIS
ENV QGIS_PREFIX_PATH=/usr
ENV QT_QPA_PLATFORM=offscreen
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

# Set the working directory in the container
WORKDIR /app

# Install Minizinc
RUN wget https://github.com/MiniZinc/MiniZincIDE/releases/download/2.8.4/MiniZincIDE-2.8.4-bundle-linux-x86_64.tgz &&\
tar -xf MiniZincIDE-2.8.4-bundle-linux-x86_64.tgz &&\
mv MiniZincIDE-2.8.4-bundle-linux-x86_64 minizinc &&\
cp -r minizinc/bin/* /usr/local/bin &&\
cp -r minizinc/lib/* /usr/local/lib &&\
cp -r minizinc/share/* /usr/local/share &&\
cp -r minizinc/lib/* /usr/local/lib

# Copy the current directory contents into the container at /app
COPY . /app

# aggiungo python3-networkx
RUN apt-get update && \
    apt-get install -y python3-pip python3-networkx wget unzip && \
    pip3 install --break-system-packages -r requirements.txt


####################################
# RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#        python3-pip \
#        python3-networkx \
#        wget \
#        unzip && \
#    pip3 install --break-system-packages -r requirements.txt && \
#    rm -rf /var/lib/apt/lists/*

# Make the pipeline script executable
COPY run_model /usr/local/bin/run_model
COPY data_processing /usr/local/bin/data_processing
COPY grid_creation /usr/local/bin/grid_creation
COPY eval_metrics /usr/local/bin/eval_metrics

RUN sed -i 's/\r$//' /usr/local/bin/run_model &&\
chmod +x /usr/local/bin/run_model
RUN sed -i 's/\r$//' /usr/local/bin/data_processing &&\
chmod +x /usr/local/bin/data_processing
RUN sed -i 's/\r$//' /usr/local/bin/grid_creation &&\
chmod +x /usr/local/bin/grid_creation
RUN sed -i 's/\r$//' /usr/local/bin/eval_metrics &&\
chmod +x /usr/local/bin/eval_metrics

# Keep the container alive
CMD ["tail", "-f", "/dev/null"]