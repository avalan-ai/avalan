# Development

Contributor setup is summarized in the root [README.md](../README.md). This
page keeps the project maintenance commands that do not belong in the main
documentation index.

## Releasing

You'll need the [GitHub CLI](https://github.com/cli/cli) for publishing
versions. On macOS, install it with `brew install gh`, then authenticate with
`gh auth login`.

Ensure you have the poetry-dynamic-versioning plugin:

```bash
poetry self add "poetry-dynamic-versioning[plugin]"
```

Create the branch for the release:

```bash
git checkout -b release/vX.Y.Z
```

Patch new version, adjusting to `minor` or `major` as appropriate:

```bash
poetry version patch
```

Commit the version patch:

```bash
git add pyproject.toml
git commit -m "Bumping version to vX.Y.Z"
```

Push the release branch:

```bash
git push -u origin release/vX.Y.Z
```

Create the pull request:

```bash
gh pr create --fill --base main --head release/vX.Y.Z
```

Once the pull request is merged, pull changes and release version X.Y.Z:

```bash
git checkout main
git pull --rebase
git tag vX.Y.Z -m "Release vX.Y.Z"
git push origin --follow-tags
```

Publish to PyPI:

```bash
poetry publish --build
```

Add the release to GitHub:

```bash
gh release create vX.Y.Z \
  --title "vX.Y.Z" \
  --notes-file <(git log --format=%B -n1 vX.Y.Z)
```

## Running Tests

Run the supported contributor dependency set with:

```bash
make test
```

## Translations

If translated strings are added with `_()` or `_n()`, update the gettext
template:

```bash
find src/avalan/. -name "*.py" | xargs xgettext \
    --language=Python \
    --keyword=_ \
    --keyword=_n \
    --package-name 'avalan' \
    --package-version `cat src/avalan/VERSION.txt` \
    --output=locale/avalan.pot
```

If you are translating to a new language such as `es`, create the folder
structure first:

```bash
mkdir -p locale/es/LC_MESSAGES
```

Update the existing `es` translation file with changes:

```bash
msgmerge --update locale/es/LC_MESSAGES/avalan.po locale/avalan.pot
```

If the `es` translation file does not exist, create it:

```bash
msginit --locale=es \
        --input=locale/avalan.pot \
        --output=locale/es/LC_MESSAGES/avalan.po
```

Edit `locale/es/LC_MESSAGES/avalan.po`, then compile it:

```bash
msgfmt --output-file=locale/es/LC_MESSAGES/avalan.mo \
       locale/es/LC_MESSAGES/avalan.po
```

## Terminal Recordings

If you are recording CLI usage and want to share it in documentation, save it
as a 480p MOV file, then generate a palette before conversion:

```bash
ffmpeg -i recording.mov \
    -vf "fps=2,scale=480:-1:flags=lanczos,palettegen" \
    /tmp/recording_palette.png
```

Convert the MOV recording to GIF with the generated palette:

```bash
ffmpeg -i recording.mov \
    -i /tmp/recording_palette.png \
    -filter_complex "fps=2,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse" \
    docs/images/recording.gif
```
