# Setup IoT Device

Ensure that the `${}` variable is properly configured before proceeding:

| Variable | Description |
|---|---|
| `${IOT_DIR}` | The path of the folder to store IoT required files. |

## Flash IoT device (oe-linux)

The official document of IQ-8275 (oe-linux) is here: https://docs.qualcomm.com/doc/80-80020-281/topic/iq8-ug-update-the-sw.html for your reference.

### Prerequisite

Prepare image from https://artifacts.codelinaro.org/ui/native/qli-ci/flashable-binaries/qimpsdk/qcs8275-iq-8275-evk-pro-sku/ on the host machine.

```bash
cd ${IOT_DIR}

wget https://artifacts.codelinaro.org/artifactory/qli-ci/flashable-binaries/qimpsdk/qcs8275-iq-8275-evk-pro-sku/x86-qcom-6.6.119-QLI.1.8-Ver.1.0_qim-product-sdk-image-2.3.0.zip
unzip x86-qcom-6.6.119-QLI.1.8-Ver.1.0_qim-product-sdk-image-2.3.0.zip
```

Download QDL tool from https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_Device_Loader?osArch=X86&osDist=Debian&osType=Linux&version=2.5.0.

```bash
wget https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/tools/Qualcomm_Device_Loader/Linux/Debian/2.5.0/QDL_2.5_Linux_x64.zip
unzip QDL_2.5_Linux_x64.zip
```

Download UFS and CDT files.

```bash
wget https://artifacts.codelinaro.org/artifactory/codelinaro-le/Qualcomm_Linux/QCS8300/provision.zip
unzip provision.zip

wget https://artifacts.codelinaro.org/artifactory/codelinaro-le/Qualcomm_Linux/QCS8300/cdt/qcs8275-iq-8275-evk-pro-sku.zip
unzip qcs8275-iq-8275-evk-pro-sku.zip
```

### Set up udev rules

Create the udev rules on the host machine.

```bash
cd /etc/udev/rules.d

vim 51-qcom-usb.rules
```
Input below content and save:

```bash
SUBSYSTEMS=="usb", ATTRS{idVendor}=="05c6", ATTRS{idProduct}=="9008", MODE="0664", GROUP="plugdev"
```

Restart udev.

```bash
sudo systemctl restart udev
```

Force the device into emergency download (EDL) mode by following steps:
1. Turn off the device
2. Turn on the SW2-3 DIP switch by pushing it up
3. Turn on the device

### UFS & CDT flashing

UFS provisioning and CDT flashing by following steps:
1. Find "USB0 Type-C port" in https://docs.qualcomm.com/doc/80-80020-281/topic/iq8-ug-introduction.html
2. Connect the device through "USB0 Type-C port" to the host machine
3. Run below commands for UFS and CDT
4. Reboot the device

```bash
cd ${IOT_DIR}

${IOT_DIR}/QDL_2.5_Linux_x64/qdl --storage ufs prog_firehose_ddr.elf provision_1_3.xml
${IOT_DIR}/QDL_2.5_Linux_x64/qdl prog_firehose_ddr.elf rawprogram3.xml patch3.xml
```

### Flash Image

Flash the SAIL and image using QDL tool.

```bash
cd ${IOT_DIR}/target/qcs8275-iq-8275-evk-pro-sku/qcom-multimedia-image/sail_nor
${IOT_DIR}/QDL_2.5_Linux_x64/qdl --storage spinor prog_firehose_ddr.elf rawprogram0.xml patch0.xml

cd ${IOT_DIR}/target/qcs8275-iq-8275-evk-pro-sku/qcom-multimedia-image
${IOT_DIR}/QDL_2.5_Linux_x64/qdl --storage ufs prog_firehose_ddr.elf rawprogram*.xml patch*.xml
```

### Configure device

Restart and use the device by following steps:
1. Turn off the device
2. Turn off the SW2-3 DIP switch
3. Turn on the device
4. Connect it with adb through "Micro-USB2 port" to the host machine, please check https://docs.qualcomm.com/doc/80-80020-281/topic/iq8-ug-introduction.html to find the "Micro-USB2 port"

---

## Flash IoT device (Ubuntu)

The official document of IQ-8275 (Ubuntu) is here: https://docs.qualcomm.com/doc/80-90441-351/topic/integrate-and-flash-software.html for your reference.

### Prerequisite

Prepare all required files for the Ubuntu image in a new directory.

```bash
cd ${IOT_DIR}
mkdir image

wget https://artifacts.codelinaro.org/artifactory/qli-ci/flashable-binaries/ubuntu-fw/QCS8300/QLI.1.7-Ver.1.1/QLI.1.7-Ver.1.1-ubuntu-QCS8300-nhlos-bins.tar.gz
tar -xvzf ./QLI.1.7-Ver.1.1-ubuntu-QCS8300-nhlos-bins.tar.gz
cp -r ./QLI.1.7-Ver.1.1-ubuntu-QCS8300-nhlos-bins/* ./image/

wget https://people.canonical.com/~platform/images/qualcomm-iot/ubuntu-24.04/ubuntu-24.04-x08/ubuntu-desktop-24.04/dtb.bin
mv ./dtb.bin ./image/

wget https://people.canonical.com/~platform/images/qualcomm-iot/ubuntu-24.04/ubuntu-24.04-x08/ubuntu-desktop-24.04/iot-qualcomm-dragonwing-classic-desktop-2404-x08-20260210.4096b.img.xz
unxz ./iot-qualcomm-dragonwing-classic-desktop-2404-x08-20260210.4096b.img.xz
mv ./iot-qualcomm-dragonwing-classic-desktop-2404-x08-20260210.4096b.img ./image/

wget https://people.canonical.com/~platform/images/qualcomm-iot/ubuntu-24.04/ubuntu-24.04-x08/ubuntu-desktop-24.04/rawprogram0.xml
mv ./rawprogram0.xml ./image/
```

Download QDL tool from https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_Device_Loader?osArch=X86&osDist=Debian&osType=Linux&version=2.5.0.

```bash
wget https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/tools/Qualcomm_Device_Loader/Linux/Debian/2.5.0/QDL_2.5_Linux_x64.zip
unzip QDL_2.5_Linux_x64.zip
```

Download UFS and CDT files.

```bash
wget https://artifacts.codelinaro.org/artifactory/codelinaro-le/Qualcomm_Linux/QCS8300/provision.zip
unzip provision.zip

wget https://artifacts.codelinaro.org/artifactory/codelinaro-le/Qualcomm_Linux/QCS8300/cdt/qcs8275-iq-8275-evk-pro-sku.zip
unzip qcs8275-iq-8275-evk-pro-sku.zip
```

### Set up udev rules

Create the udev rules on the host machine.

```bash
cd /etc/udev/rules.d

vim 51-qcom-usb.rules
```
Input below content and save:

```bash
SUBSYSTEMS=="usb", ATTRS{idVendor}=="05c6", ATTRS{idProduct}=="9008", MODE="0664", GROUP="plugdev"
```

Restart udev.

```bash
sudo systemctl restart udev
```

Force the device into emergency download (EDL) mode by following steps:
1. Turn off the device
2. Turn on the SW2-3 DIP switch by pushing it up
3. Turn on the device

### UFS & CDT flashing

UFS provisioning and CDT flashing by following steps:
1. Find "USB0 Type-C port" in https://docs.qualcomm.com/doc/80-80020-281/topic/iq8-ug-introduction.html
2. Connect the device through "USB0 Type-C port" to the host machine
3. Run below commands for UFS and CDT
4. Reboot the device

```bash
cd ${IOT_DIR}

${IOT_DIR}/QDL_2.5_Linux_x64/qdl --storage ufs prog_firehose_ddr.elf provision_1_3.xml
${IOT_DIR}/QDL_2.5_Linux_x64/qdl prog_firehose_ddr.elf rawprogram3.xml patch3.xml
```

### Flash Image

Flash the SAIL and image using QDL tool.

```bash
cd ${IOT_DIR}/image/sail_nor
${IOT_DIR}/QDL_2.5_Linux_x64/qdl --storage spinor prog_firehose_ddr.elf rawprogram0.xml patch0.xml

cd ${IOT_DIR}/image/
${IOT_DIR}/QDL_2.5_Linux_x64/qdl --storage ufs prog_firehose_ddr.elf rawprogram*.xml patch*.xml
```

### Configure device

Restart and use the device by following steps:
1. Turn off the device
2. Turn off the SW2-3 DIP switch
3. Turn on the device
4. Connect it with adb through "Micro-USB2 port" to the host machine, please check https://docs.qualcomm.com/doc/80-80020-281/topic/iq8-ug-introduction.html to find the "Micro-USB2 port"
5. Sign in, the default username and password are both "ubuntu"

Configure Ubuntu on the device, the official document is here: https://docs.qualcomm.com/doc/80-90441-351/topic/use-ubuntu-on-iq8.html.

```bash
sudo apt-add-repository -s ppa:ubuntu-qcom-iot/qcom-ppa
sudo apt update && sudo apt upgrade
sudo apt install libatomic1
sudo apt install libqnn1 qnn-tools libqnn-dev
```

You should find libcdsprpc.so and libdmabufheap.so.0 in:
- `/usr/lib/aarch64-linux-gnu/libcdsprpc.so`
- `/usr/lib/aarch64-linux-gnu/libdmabufheap.so.0`
