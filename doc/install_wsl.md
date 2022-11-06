# Install Helpers

# wsl

If you haven't used wsl for a while..., and [forgotten your passwd](https://askubuntu.com/questions/931940/unable-to-change-the-root-password-in-windows-10-wsl)!

Run the follow commands if there are issues installing [python3](https://www.itsupportwale.com/blog/how-to-upgrade-to-python-3-8-on-ubuntu-18-04-lts/), venv, etc before creating the venv:

```bash
sudo hwclock --hctosys

sudo apt update

sudo apt upgrade
```

```bash
sudo apt-get install python3.8
```

but follow [this python3 install](https://www.itsupportwale.com/blog/how-to-upgrade-to-python-3-8-on-ubuntu-18-04-lts/) to make sure set up is correct"

Then:

```bash
sudo apt-get install python3.8-venv

sudo apt install python3-pip

python -m pip install --upgrade pip need to do this in venv
```

close windoe, oen new ubuntu-18.04

Th pip install [upgrade](https://stackoverflow.com/questions/64517366/python-error-while-installing-matplotlib) is important and must be done in the venv.

BUT

sudo update-alternatives --config python3

switch back to python3.6

do:

sudo apt update

sudo apt-get update

switch to python3.8 (select the auto version)

THEN create venv! If that works, check your --versions and it should be OK to procede with installing the requirements. 