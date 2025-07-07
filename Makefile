REAL_TARGETS := install lint test test-coverage version release

ifneq ($(filter-out $(REAL_TARGETS),$(MAKECMDGOALS)),)
.PHONY: $(filter-out $(REAL_TARGETS),$(MAKECMDGOALS))
$(filter-out $(REAL_TARGETS),$(MAKECMDGOALS)):
	@:
endif

install:
	poetry sync --extras all

lint:
	poetry run ruff format --preview src/ tests/
	poetry run black --preview --enable-unstable-feature=string_processing src/ tests/
	poetry run ruff check --fix src/ tests/

test:
	poetry sync --extras test
	poetry run pytest --verbose

test-coverage:
	$(eval ARGS := $(filter-out $@,$(MAKECMDGOALS)))
	$(eval COVERAGE_THRESHOLD := $(firstword $(ARGS)))
	$(eval COVERAGE_PATH := $(if $(word 2,$(ARGS)),$(word 2,$(ARGS)),src/))
	@poetry run pytest --cov=$(COVERAGE_PATH) --cov-report=json &> /dev/null
	@if [ -z "$(COVERAGE_THRESHOLD)" ]; then \
		jq -r '.files | to_entries[] | "\(.key): \(.value.summary.percent_covered_display)%"' coverage.json; \
	elif [ "$(COVERAGE_THRESHOLD)" -lt 0 ]; then \
		jq -r --arg thr "$$(echo $(COVERAGE_THRESHOLD) | sed 's/^-//')" '.files | to_entries[] | select(.value.summary.percent_covered < ($$thr|tonumber)) | "\(.key): \(.value.summary.percent_covered_display)%"' coverage.json; \
	else \
		jq -r --arg thr "$(COVERAGE_THRESHOLD)" '.files | to_entries[] | select(.value.summary.percent_covered >= ($$thr|tonumber)) | "\(.key): \(.value.summary.percent_covered_display)%"' coverage.json; \
	fi

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
	@$(eval NOTES := $(shell git log --format=%B -n1 v$(VERSION)))
	git push origin --follow-tags
	poetry publish --build
	gh release create v$(VERSION) \
	  --title "v$(VERSION)" \
	  --notes "$(NOTES)"

%:
	@:
