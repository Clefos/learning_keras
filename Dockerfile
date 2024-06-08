FROM rocm/tensorflow:latest

RUN sudo apt-get update -y

RUN sudo python -m pip install --upgrade pip

RUN sudo pip3 install pillow
RUN sudo pip3 install matplotlib

# Prevent tensorflow from taking all gpu memory at once
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

ENV TF_ENABLE_AUTO_MIXED_PRECISION=1

#ENV CUDA_VISIBLE_DEVICES=-1

# Same as in base image rocm/tensorflow
#CMD ["/bin/bash", "-c", "env > /etc/profile.d/horovod.sh"]
CMD ["/bin/bash"]
