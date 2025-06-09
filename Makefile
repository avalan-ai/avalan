.PHONY: install lint release test version

install:
	poetry sync --extras all

lint:
	poetry run ruff format --preview src/ tests/
	poetry run black --preview --enable-unstable-feature=string_processing src/ tests/
	poetry run ruff check --fix src/ tests/

test:
	poetry sync --extras test
	poetry run pytest --verbose -s

version:
	$(eval VERSION := $(filter-out $@,$(MAKECMDGOALS)))
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make version X.Y.Z"; \
		exit 1; \
	fi
	git checkout main
	git pull --rebase
	git checkout -b "release/v$(VERSION)"
	poetry version "$(VERSION)"
	git add pyproject.toml
	git commit -m "Bumping version to v$(VERSION)"
	git push -u origin "release/v$(VERSION)"

release:
	$(eval VERSION := $(filter-out $@,$(MAKECMDGOALS)))
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make release X.Y.Z"; \
		exit 1; \
	fi
	git checkout main
	git pull --rebase
	git tag v$(VERSION) -m "Release v$(VERSION)"
	git push origin --follow-tags
	poetry publish --build
	gh release create v$(VERSION) \
	  --title "v$(VERSION)" \
	  --notes-file <(git log --format=%B -n1 "v$(VERSION)")

%:
	@:
