.PHONY: install lint test release

install:
	poetry sync --extras all

lint:
	poetry run ruff check --fix
	poetry run ruff format

test:
	poetry sync --extras test
	poetry run pytest --verbose -s

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
	  --notes-file <(git log --format=%B -n1 v$(VERSION))

%:
	@:
