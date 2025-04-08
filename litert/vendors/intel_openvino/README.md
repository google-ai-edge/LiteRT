# INTEL OpenVINO™ DELEGATE FOR LiteRT FRAMEWORK

## System Requirement<br>

The build has been verified with the following system requirements:<br>

*	Ubuntu 22.04 or Ubuntu 24.04<br>
*	Bazel 7.4.1<br>
*	Clang 17.0.2<br>


## Build From Source with Bazel for Linux Systems

1. From your working directory run
	```
	git clone https://github.com/intel-innersource/os.android.bsp.litert
	git submodule init && git submodule update --remote
	```

2. Setup Intel OpenVINO™  for LiteRT

	In your working directory run the following setps. Obtain the correct OpenVINO™ release package based on your Linux Build environment(Ubuntu 22.xx or Ubuntu 24.xx)

	```
	For Ubuntu 22.04:
	wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.1/linux/openvino_toolkit_ubuntu22_2025.1.0.18503.6fec06580ab_x86_64.tgz
	For Ubuntu 24.xx:
	wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.1/linux/openvino_toolkit_ubuntu24_2025.1.0.18503.6fec06580ab_x86_64.tgz
	```

	Untar the file. Replace xx in below commands with the version that was downloaded
	```
	tar -xf openvino_toolkit_ubuntu<xx>_2025.1.0.18503.6fec06580ab_x86_64.tgz
	mkdir -p /opt/intel
	mv openvino_toolkit_ubuntu<xx>_2025.1.0.18503.6fec06580ab_x86_64.tgz /opt/intel/openvino_2025.1.0
	cd /opt/intel/openvino_2025.1.0/
	./install_dependencies/install_openvino_dependencies.sh
	cd /opt/intel/
	sudo ln -s openvino_2025.1.0 openvino_2025
	export OPENVINO_NATIVE_DIR=/opt/intel/openvino_2025 
	```

3. You will need docker, but nothing else. Create the image with

    ```
    docker build . -t tflite-builder -f ci/tflite-py3.Dockerfile

    # Confirm the container was created with
    docker image ls
    ```

4. Run bash inside the container
	Pass your machine's /opt/intel path so that container gets the Intel OpenVINO™ installation
    ```
    docker run -it -w /host_dir -v $PWD:/host_dir -v $HOME/.cache/bazel:/root/.cache/bazel -v /opt:/opt tflite-builder bash
    ```

	where -v /opt:/opt allows the container to use the Intel OpenVINO™ installed in step 2<br>
    where `-v $HOME/.cache/bazel:/root/.cache/bazel` is optional, but useful to
    map your Bazel cache into the container.

5. Build Command:
	Inside the docker goto your host_dir and use the build command
	```
	export OPENVINO_NATIVE_DIR=/opt/intel/openvino_2025 
	cd /host_dir
	CC=clang bazel build //litert/vendors/intel_openvino/...
	```

## Build From Source with Bazel for Android Systems<br>
<br>
	To be done
