REAL_TARGETS := install lint test tests test-pgsql tests-pgsql test-coverage test-coverage-exact test-pgsql-exact typecheck-input-contract version release
TEST_ARGS := $(filter-out $(REAL_TARGETS),$(MAKECMDGOALS))
PYTEST_ARGS := --verbose
TASK_PGSQL_TEST_DEPS := "alembic>=1.17.2,<2.0.0"
INPUT_CONTRACT_SCRIPTS := scripts/input_contract_json.py scripts/run_input_contract_gate.py scripts/task_pgsql_test_database.py scripts/verify_input_acceptance.py scripts/verify_input_types.py scripts/verify_src_coverage.py
LINT_PATHS := src/ tests/ $(INPUT_CONTRACT_SCRIPTS)

ifneq ($(filter coverage coverage-report,$(TEST_ARGS)),)
PYTEST_ARGS += --cov=src/ --cov-report=xml
endif

ifneq ($(filter coverage,$(TEST_ARGS)),)
# coverage.py treats fail-under=100 as exact; enforce reported 100.00%.
PYTEST_ARGS += --cov-fail-under=99.995 --cov-precision=2
endif

ifneq ($(filter-out $(REAL_TARGETS),$(MAKECMDGOALS)),)
.PHONY: $(filter-out $(REAL_TARGETS),$(MAKECMDGOALS))
$(filter-out $(REAL_TARGETS),$(MAKECMDGOALS)):
	@:
endif

install:
	poetry sync --all-extras

lint:
	poetry run ruff format --preview $(LINT_PATHS)
	poetry run black --preview --enable-unstable-feature=string_processing $(LINT_PATHS)
	poetry run ruff check --fix $(LINT_PATHS)
	poetry run mypy
	poetry run mypy $(INPUT_CONTRACT_SCRIPTS)

test:
ifeq ($(filter no-install,$(TEST_ARGS)),)
	poetry sync --all-extras --with test
endif
ifneq ($(strip $(AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN)),)
	poetry run python -m pip install $(TASK_PGSQL_TEST_DEPS)
	poetry run -- python scripts/task_pgsql_test_database.py -- $(PYTEST_ARGS)
else ifneq ($(strip $(AVALAN_TASK_TEST_POSTGRESQL_DOCKER)),)
	poetry run python -m pip install $(TASK_PGSQL_TEST_DEPS)
	poetry run -- python scripts/task_pgsql_test_database.py --docker -- $(PYTEST_ARGS)
else
	poetry run pytest $(PYTEST_ARGS)
endif

.PHONY: test tests test-pgsql tests-pgsql test-coverage-exact test-pgsql-exact typecheck-input-contract
tests: test

test-pgsql:
ifeq ($(filter no-install,$(TEST_ARGS)),)
	poetry sync --all-extras --with test
endif
	poetry run python -m pip install $(TASK_PGSQL_TEST_DEPS)
	poetry run -- python scripts/task_pgsql_test_database.py --docker -- $(PYTEST_ARGS)

tests-pgsql: test-pgsql

test-coverage-exact:
ifeq ($(filter no-install,$(TEST_ARGS)),)
	poetry sync --all-extras --with test
endif
	poetry run python scripts/run_input_contract_gate.py --coverage-only

test-pgsql-exact:
	@test -n "$(INPUT_PHASE)" || (echo "INPUT_PHASE is required" >&2; exit 2)
ifeq ($(filter no-install,$(TEST_ARGS)),)
	poetry sync --all-extras --with test
endif
	poetry run python -m pip install $(TASK_PGSQL_TEST_DEPS)
	poetry run -- python scripts/task_pgsql_test_database.py --docker --runner-script scripts/run_input_contract_gate.py -- --through-phase $(INPUT_PHASE)

typecheck-input-contract:
	@test -n "$(INPUT_PHASE)" || (echo "INPUT_PHASE is required" >&2; exit 2)
	poetry run python scripts/verify_input_types.py --through-phase $(INPUT_PHASE)

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
	poetry build --clean
	poetry publish
	gh release create v$(VERSION) \
	  --title "v$(VERSION)" \
	  --notes "$(NOTES)"

%:
	@:
