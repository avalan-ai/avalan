# Installation

## MacOS

Install avalan using [Homebrew](https://brew.sh):

```bash
brew install avalan
```

## Ubuntu

Update package index and install python prerequisites:

```bash
sudo apt update -y
sudo apt install -y python3 python3-venv python3-dev python3-pip curl
```

Install [poetry](https://python-poetry.org) and add it to `$PATH`:

```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Start a new poetry project, specify the python version, and add avalan with
`all` extras:

```bash
mkdir avalan-test/ && cd avalan-test/
poetry init --no-interaction --python=">=3.12,<3.14"
poetry add "avalan[all]" --no-cache
```
