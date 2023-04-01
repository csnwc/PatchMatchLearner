# jetson ssh setup


[Getting started with Jetson Nano](https://medium.com/@heldenkombinat/getting-started-with-the-jetson-nano-37af65a07aab#:~:text=1%20Flash%20the%20OS%20and%20boot%20up.%20The,environments.%20...%207%20Setting%20up%20Jupyter%20Notebook.%20)

```bash
PS C:\Users\daveg> ssh daveg@192.168.1.195
The authenticity of host '192.168.1.195 (192.168.1.195)' can't be established.
ED25519 key fingerprint is SHA256:RAaQygPBoZkR78rLirbNQijjkCZzi/glEpfIrUITZ7A.
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added '192.168.1.195' (ED25519) to the list of known hosts.
daveg@192.168.1.195's password:
Welcome to Ubuntu 20.04.5 LTS (GNU/Linux 4.9.299-tegra aarch64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage
This system has been minimized by removing packages and content that are
not required on a system that users do not log into.

To restore this content, you can run the 'unminimize' command.

 * Introducing Expanded Security Maintenance for Applications.
   Receive updates to over 25,000 software packages with your
   Ubuntu Pro subscription. Free for personal use.

     https://ubuntu.com/pro

Expanded Security Maintenance for Applications is not enabled.

2 updates can be applied immediately.
2 of these updates are standard security updates.
To see these additional updates run: apt list --upgradable

37 additional security updates can be applied with ESM Apps.
Learn more about enabling ESM Apps service at https://ubuntu.com/esm


The programs included with the Ubuntu system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Ubuntu comes with ABSOLUTELY NO WARRANTY, to the extent permitted by
applicable law.

daveg@daveg-desktop:~$ l
```

## J101 Enable SD Card

[J101 Enable SD Card](https://wiki.seeedstudio.com/J101_Enable_SD_Card/#:~:text=Driver%20Configuration%201%20Step%201%20.%20Clone%20the,Name%20our%20device%20and%20Finish%20driver%20installation%20)

```bash
daveg@daveg-desktop:~/tmp$ git clone https://github.com/Seeed-Studio/seeed-linux-dtoverlays.git
Cloning into 'seeed-linux-dtoverlays'...
remote: Enumerating objects: 2879, done.
remote: Counting objects: 100% (449/449), done.
remote: Compressing objects: 100% (164/164), done.
remote: Total 2879 (delta 341), reused 370 (delta 285), pack-reused 2430
Receiving objects: 100% (2879/2879), 3.41 MiB | 5.84 MiB/s, done.
Resolving deltas: 100% (1541/1541), done.
daveg@daveg-desktop:~/tmp$ l
total 4
drwxrwxr-x 13 daveg daveg 4096 Mar 13 20:58 seeed-linux-dtoverlays/
daveg@daveg-desktop:~/tmp$ cd seeed-linux-dtoverlays/
daveg@daveg-desktop:~/tmp/seeed-linux-dtoverlays$ sed -i '17s#JETSON_COMPATIBLE#\"nvidia,p3449-0000-b00+p3448-0002-b00\"\, \"nvidia\,jetson-nano\"\, \"nvidia\,tegra210\"#' overlays/jetsonnano/jetson-sdmmc-overlay.dts
daveg@daveg-desktop:~/tmp/seeed-linux-dtoverlays$ make overlays/jetsonnano/jetson-sdmmc-overlay.dtbo
  DTC     overlays/jetsonnano/jetson-sdmmc-overlay.dtbo
daveg@daveg-desktop:~/tmp/seeed-linux-dtoverlays$ sudo cp overlays/jetsonnano/jetson-sdmmc-overlay.dtbo /boot/
[sudo] password for daveg:
daveg@daveg-desktop:~/tmp/seeed-linux-dtoverlays$ cd /boot/
daveg@daveg-desktop:/boot$ sudo /opt/nvidia/jetson-io/config-by-hardware.py -l
Header 1 [default]: Jetson 40pin Header
  Available hardware modules:
  1. reComputer sdmmc
Header 2: Jetson Nano CSI Connector
  Available hardware modules:
  1. Camera IMX219 Dual
  2. Camera IMX477 Dual
  3. Camera IMX477-A and IMX219-B
daveg@daveg-desktop:/boot$ sudo /opt/nvidia/jetson-io/config-by-hardware.py -n "reComputer sdmmc"
Configuration saved to /boot/kernel_tegra210-p3448-0002-p3449-0000-b00-user-custom.dtb.
Reboot system to reconfigure.
daveg@daveg-desktop:/boot$ sudo reboot
Connection to 192.168.1.195 closed by remote host.
Connection to 192.168.1.195 closed.
PS C:\Users\daveg>
```

## Qengineering solution: Solved

The main point is that the seed studio board is different from all the examples. Lots of sltions don't work for their board or the instructions don't fit! Do the following, be careful of short cuts:

  1) Follow: [Qe Ubuntu20.04 image](https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image)

  2) [Boot from Qe image on usb](https://github.com/jetsonhacks/bootFromUSB). KEY POINT; edit */boot/extlinux/extlinux.conf* *ON seed studio mmblk0 drive*, the hardware encoded primary!!! Used [£58 Western Digital 2TB External Hard Drive: ](https://amzn.to/3t7A7jH). Partion 250GB for system, 1700GB for data!

  3) [Manual install of pytorch and torchvision](https://qengineering.eu/install-pytorch-on-jetson-nano.html). Couldn't find torch/torchvision on USB image, suspect it is hidden in a jetson user account and the image assumes that the dst is a genuine NVIDIA jetson dev board. Easy to install from Qe globally but follow below for local venv version. Note: torch-1.12 with torchvision-0.13.

```bash
  417  python3
  418  source venv/bin/activate
  419  pip3 install wheel
  420  pip3 install torch-1.12.0a0+git67ece03-cp38-cp38-linux_aarch64.whl
  421  pip3 freeze
  422  pip freeze
  423  pip3 install future
  424  pip freeze
  425  pip3 freeze
  426  pip3 install -U --user wheel mock pillow
  427  pip3 install mock pillow
  428  pip3 freeze
  429  pip3 install testresources
  430  pip3 freeze
  431  pip3 install setuptools==58.3.0
  432  pip3 install Cython
  433  pip3 install gdown
  434  pip3 freeze
  435  gdown https://drive.google.com/uc?id=1MnVB7I4N8iVDAkogJO76CiQ2KRbyXH_e
  436  pip3 install torch-1.12.0a0+git67ece03-cp38-cp38-linux_aarch64.whl
  437  python3
  438  l
  439  rm torch-1.12.0a0+git67ece03-cp38-cp38-linux_aarch64.whl
  440  pip3 install -U pillow
  441  gdown https://drive.google.com/uc?id=11DPKcWzLjZa5kRXRodRJ3t9md0EMydhj
  442  l
  443  pip3 install torchvision-0.13.0a0+da3794e-cp38-cp38-linux_aarch64.whl
  444  rm torchvision-0.13.0a0+da3794e-cp38-cp38-linux_aarch64.whl
  445  python3
  446  h
  447  history
```

```bash
For shutdown:

sudo poweroff
For restart:

sudo reboot
```

### LightningAI:

```bash
python -m pip install lightning
```

Most OK but get:

```bash
ERROR: fastapi 0.88.0 has requirement starlette==0.22.0, but you'll have starlette 0.26.1 which is incompatible.
```

```bash
pip install lightning-bolts
```

Some difficulties, did, as suggested:

```bash
pip install lightning[extra]
```

Got:

```bash
ERROR: s3fs 2022.11.0 has requirement fsspec==2022.11.0, but you'll have fsspec 2023.3.0 which is incompatible.
```

Fixed a load of old versus new LightningAI conflicts.

Need to create a jetson [branch and push](https://stackoverflow.com/questions/52231262/how-to-push-a-new-branch-in-git)

Install:

```bash
https://stackoverflow.com/questions/52231262/how-to-push-a-new-branch-in-git
```

In:

```bash
pytorch-lightning/loggers/__init__.py
```

```bash
from pytorch_lightning.loggers.logger import Logger
import pytorch_lightning.loggers.logger as LightningLoggerBase ## **DG**
# from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase ## **DG**
```

Then you get [from pytorch_lightning.utilities.apply_func import apply_to_collection
ModuleNotFoundError: No module named 'pytorch_lightning.utilities.apply_func'](https://github.com/Lightning-Universe/lightning-flash/issues/1539)

Then fix two more similar.

Then

```bash
pip3 install -U scikit-learn
```

Still get lots of warnings but it works.

[Checkout and push to a new branch](https://stackoverflow.com/questions/52231262/how-to-push-a-new-branch-in-git)


